import os
import json
import math
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

class ContrastiveDatasetGenerator:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the input CSV file."""
        df = pd.read_csv(self.input_file)
        tables = df.copy()
        paragraphs = df.copy()
        return tables, paragraphs
    
    def get_distance(self, p1: pd.Series, p2: pd.Series) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt(
            (p1['X position'] - p2['X position'])**2 + 
            (p1['Y position'] - p2['Y position'])**2
        )
    
    def parse_table_content(self, table_str: str) -> List[List[str]]:
        """Parse table content from string to nested list."""
        try:
            return json.loads(table_str)
        except Exception:
            return table_str

    def parse_sentence_context(self, sentence_context_str: str) -> List[str]:
        """Parse sentence context from string to list."""
        try:
            return json.loads(sentence_context_str)
        except Exception:
            return sentence_context_str

    def sample_additional_positives(self, topic_paragraphs: pd.DataFrame, table: pd.Series, 
                                  table_id: int, max_samples: int = 9) -> List[Dict[str, Any]]:
        """Sample a limited number of additional positives."""
        # Exclude the primary positive
        other_paragraphs = topic_paragraphs[topic_paragraphs['id'] != table_id]
        
        # Calculate distances for all potential positives
        distances = [(idx, self.get_distance(table, row)) 
                    for idx, row in other_paragraphs.iterrows()]
        
        # Sort by distance and take top max_samples
        distances.sort(key=lambda x: x[1])
        selected_indices = [idx for idx, _ in distances[:max_samples]]
        
        additional_positives = []
        for idx in selected_indices:
            para = other_paragraphs.loc[idx]
            additional_positives.append({
                'id': int(para['id']),
                'distance': float(self.get_distance(table, para)),
                'broad_topic': para['Nomic Topic: broad'],
                'medium_topic': para['Nomic Topic: medium'],
                'sentence_context': self.parse_sentence_context(para['sentence_context'])  # Updated here
            })
            
        return additional_positives

    def sample_negatives(self, anchor: pd.Series, topic_to_exclude: str, 
                        paragraphs: pd.DataFrame, threshold: float,
                        num_hard: int = 5, num_extreme: int = 5) -> List[Dict[str, Any]]:
        """Sample negatives based primarily on topic relationships, with distance as validation."""
        negatives = []
        
        # Use provided threshold
        distance_threshold = threshold
        
        # Get all paragraphs grouped by topic
        topic_groups = paragraphs.groupby(['Nomic Topic: broad', 'Nomic Topic: medium'])
        
        # Sample Hard Negatives
        hard_negatives = []
        
        # First try: Get from different medium topics within same broad topic
        same_broad_groups = [
            (broad, medium, group) 
            for (broad, medium), group in topic_groups
            if broad == topic_to_exclude and medium != anchor['Nomic Topic: medium']
        ]
        
        for broad, medium, group in same_broad_groups:
            if len(hard_negatives) >= num_hard:
                break
                
            # Sample one from this medium topic
            if not group.empty:
                sample = group.sample(n=1).iloc[0]
                distance = self.get_distance(anchor, sample)
                
                # Check distance threshold
                if distance <= distance_threshold:
                    hard_negatives.append({
                        'negative_id': int(sample['id']),
                        'distance': float(distance),
                        'broad_topic': sample['Nomic Topic: broad'],
                        'medium_topic': sample['Nomic Topic: medium'],
                        'sentence_context': self.parse_sentence_context(sample['sentence_context'])  # Updated here
                    })
        
        # If we need more, take more samples from closest topic
        if len(hard_negatives) < num_hard and same_broad_groups:
            closest_medium = same_broad_groups[0][1]  # Take first medium topic
            remaining_needed = num_hard - len(hard_negatives)
            
            closest_group = topic_groups.get_group((topic_to_exclude, closest_medium))
            if not closest_group.empty:
                additional_samples = closest_group.sample(n=min(remaining_needed, len(closest_group)))
                
                for _, sample in additional_samples.iterrows():
                    distance = self.get_distance(anchor, sample)
                    if distance <= distance_threshold:
                        hard_negatives.append({
                            'negative_id': int(sample['id']),
                            'distance': float(distance),
                            'broad_topic': sample['Nomic Topic: broad'],
                            'medium_topic': sample['Nomic Topic: medium'],
                            'sentence_context': self.parse_sentence_context(sample['sentence_context'])  # Updated here
                        })
        
        # If still insufficient, get from different broad topics
        if len(hard_negatives) < num_hard:
            other_topic_groups = [
                (broad, medium, group) 
                for (broad, medium), group in topic_groups
                if broad != topic_to_exclude
            ]
            
            # Calculate minimum distance for each group to anchor
            def get_group_min_distance(group_tuple):
                _, _, group_df = group_tuple
                if group_df.empty:
                    return float('inf')
                distances = [self.get_distance(anchor, row) for _, row in group_df.iterrows()]
                return min(distances) if distances else float('inf')
                
            # Sort groups by minimum distance to anchor
            other_topic_groups.sort(key=get_group_min_distance)
            
            for broad, medium, group in other_topic_groups:
                if len(hard_negatives) >= num_hard:
                    break
                    
                if not group.empty:
                    sample = group.sample(n=1).iloc[0]
                    distance = self.get_distance(anchor, sample)
                    
                    if distance <= distance_threshold:
                        hard_negatives.append({
                            'negative_id': int(sample['id']),
                            'distance': float(distance),
                            'broad_topic': sample['Nomic Topic: broad'],
                            'medium_topic': sample['Nomic Topic: medium'],
                            'sentence_context': self.parse_sentence_context(sample['sentence_context'])  # Updated here
                        })
        
        # Sample Extreme Negatives
        extreme_negatives = []
        used_broad_topics = set()
        
        # First try: Get from different broad topics
        other_broad_groups = [
            (broad, medium, group) 
            for (broad, medium), group in topic_groups
            if broad != topic_to_exclude
        ]
        
        for broad, medium, group in other_broad_groups:
            if len(extreme_negatives) >= num_extreme:
                break
                
            if broad not in used_broad_topics and not group.empty:
                sample = group.sample(n=1).iloc[0]
                distance = self.get_distance(anchor, sample)
                
                if distance > distance_threshold:
                    extreme_negatives.append({
                        'negative_id': int(sample['id']),
                        'distance': float(distance),
                        'broad_topic': sample['Nomic Topic: broad'],
                        'medium_topic': sample['Nomic Topic: medium'],
                        'sentence_context': self.parse_sentence_context(sample['sentence_context'])  # Updated here
                    })
                    used_broad_topics.add(broad)
        
        # If insufficient, take additional samples from different broad topics
        if len(extreme_negatives) < num_extreme:
            remaining_needed = num_extreme - len(extreme_negatives)
            
            # Create a clean copy of the filtered DataFrame
            other_samples = paragraphs[
                (paragraphs['Nomic Topic: broad'] != topic_to_exclude) &
                (~paragraphs['id'].isin([n['negative_id'] for n in extreme_negatives]))
            ].copy()
            
            if not other_samples.empty:
                # Calculate distances safely using loc
                other_samples.loc[:, 'distance'] = other_samples.apply(
                    lambda x: self.get_distance(anchor, x), axis=1
                )
                
                # Sort by distance (descending) and take top remaining_needed
                other_samples = other_samples.sort_values('distance', ascending=False)
                
                for _, sample in other_samples.head(remaining_needed).iterrows():
                    if sample['distance'] > distance_threshold:
                        extreme_negatives.append({
                            'negative_id': int(sample['id']),
                            'distance': float(sample['distance']),
                            'broad_topic': sample['Nomic Topic: broad'],
                            'medium_topic': sample['Nomic Topic: medium'],
                            'sentence_context': self.parse_sentence_context(sample['sentence_context'])  # Updated here
                        })
        
        return hard_negatives + extreme_negatives

    def generate_dataset(self) -> List[Dict[str, Any]]:
        """Generate the contrastive dataset."""
        # Load data
        tables, paragraphs = self.load_data()
        contrastive_data = []
        
        # Process each table with progress bar
        for idx, table in tqdm(tables.iterrows(), total=len(tables), desc="Processing tables"):
            table_id = int(table['id'])
            table_broad_topic = table['Nomic Topic: broad']
            
            # Get the matching paragraph
            matching_paragraph = paragraphs[paragraphs['id'] == table_id].iloc[0]
            
            # Get additional positives from same topic
            topic_paragraphs = paragraphs[
                (paragraphs['Nomic Topic: broad'] == table_broad_topic)
            ]
            
            additional_positives = self.sample_additional_positives(
                topic_paragraphs, 
                table, 
                table_id, 
                max_samples=9
            )
            
            # Calculate threshold for this table
            all_distances = paragraphs.apply(lambda x: self.get_distance(table, x), axis=1)
            max_distance = all_distances.max()
            threshold = max_distance / 3
            
            # Sample negatives with the calculated threshold
            negatives = self.sample_negatives(table, table_broad_topic, paragraphs, threshold=threshold)
            
            # Create triplet
            triplet = {
                'threshold': threshold,  # Store the threshold
                'anchor': {
                    'id': table_id,
                    'broad_topic': table['Nomic Topic: broad'],
                    'medium_topic': table['Nomic Topic: medium'],
                    'table_content': self.parse_table_content(table['table']),
                    'caption': "" if pd.isna(table['caption']) or table['caption'] == "null" else table['caption'],
                    'table_title': "" if pd.isna(table['table_title']) or table['table_title'] == "null" else table['table_title']
                },
                'primary_positive': {
                    'id': int(matching_paragraph['id']),
                    'distance': float(self.get_distance(table, matching_paragraph)),
                    'broad_topic': matching_paragraph['Nomic Topic: broad'],
                    'medium_topic': matching_paragraph['Nomic Topic: medium'],
                    'sentence_context': self.parse_sentence_context(matching_paragraph['sentence_context'])  # Updated here
                },
                'additional_positives': additional_positives,
                'negatives': negatives
            }
            
            contrastive_data.append(triplet)
        
        return contrastive_data

    def save_dataset(self, data: List[Dict[str, Any]], filename: str = 'contrastive_dataset.json'):
        """Save dataset and collect comprehensive statistics focusing on topic relationships and dynamic thresholds."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize diagnostic statistics
        diagnostic_stats = {
            'overall_stats': {
                'total_triplets': len(data),
                'total_positives': sum(len(d['additional_positives']) + 1 for d in data),
                'total_negatives': sum(len(d['negatives']) for d in data),
                'triplets_with_insufficient_negatives': sum(1 for d in data if len(d['negatives']) < 10)
            },
            'topic_distribution': {
                'broad_topics': defaultdict(int),
                'medium_topics': defaultdict(int),
                'topic_pairs': defaultdict(int)
            },
            'negative_sampling_stats': {
                'hard_negatives': {
                    'same_broad_different_medium': 0,
                    'different_broad': 0,
                    'avg_distance': 0.0,
                    'min_distance': float('inf'),
                    'max_distance': 0.0
                },
                'extreme_negatives': {
                    'unique_broad_topics': set(),  # Will be converted to count later
                    'avg_distance': 0.0,
                    'min_distance': float('inf'),
                    'max_distance': 0.0
                }
            },
            'problematic_cases': []
        }
        
        # Analyze each triplet
        for triplet in data:
            anchor_broad = triplet['anchor']['broad_topic']
            anchor_medium = triplet['anchor']['medium_topic']
            
            # Track topic distributions
            diagnostic_stats['topic_distribution']['broad_topics'][anchor_broad] += 1
            diagnostic_stats['topic_distribution']['medium_topics'][anchor_medium] += 1
            # Using string key instead of tuple for topic pairs
            topic_pair_key = f"{anchor_broad}||{anchor_medium}"
            diagnostic_stats['topic_distribution']['topic_pairs'][topic_pair_key] += 1
            
            # Calculate threshold for this triplet
            threshold = triplet.get('threshold', 0)  # Use stored threshold or default to 0
            
            # Analyze negatives
            hard_neg_sum = 0
            hard_neg_count = 0
            extreme_neg_sum = 0
            extreme_neg_count = 0
            
            for neg in triplet['negatives']:
                distance = neg['distance']
                is_hard = distance <= threshold
                
                if is_hard:
                    if neg['broad_topic'] == anchor_broad:
                        diagnostic_stats['negative_sampling_stats']['hard_negatives']['same_broad_different_medium'] += 1
                    else:
                        diagnostic_stats['negative_sampling_stats']['hard_negatives']['different_broad'] += 1
                        
                    # Update hard negative statistics
                    hard_neg_sum += distance
                    hard_neg_count += 1
                    hard_neg_stats = diagnostic_stats['negative_sampling_stats']['hard_negatives']
                    hard_neg_stats['min_distance'] = min(hard_neg_stats['min_distance'], distance)
                    hard_neg_stats['max_distance'] = max(hard_neg_stats['max_distance'], distance)
                else:
                    # Update extreme negative statistics
                    extreme_neg_sum += distance
                    extreme_neg_count += 1
                    extreme_neg_stats = diagnostic_stats['negative_sampling_stats']['extreme_negatives']
                    extreme_neg_stats['min_distance'] = min(extreme_neg_stats['min_distance'], distance)
                    extreme_neg_stats['max_distance'] = max(extreme_neg_stats['max_distance'], distance)
                    extreme_neg_stats['unique_broad_topics'].add(neg['broad_topic'])
        
        # Process the statistics for serialization
        hard_neg_stats = diagnostic_stats['negative_sampling_stats']['hard_negatives']
        extreme_neg_stats = diagnostic_stats['negative_sampling_stats']['extreme_negatives']
        
        # Calculate averages
        if hard_neg_count > 0:
            hard_neg_stats['avg_distance'] = hard_neg_sum / hard_neg_count
        if extreme_neg_count > 0:
            extreme_neg_stats['avg_distance'] = extreme_neg_sum / extreme_neg_count
        
        # Convert set size to count for unique broad topics and remove the set
        extreme_neg_stats['unique_broad_topics'] = len(extreme_neg_stats['unique_broad_topics'])
        
        # Remove any infinity values if no negatives were found
        if hard_neg_stats['min_distance'] == float('inf'):
            hard_neg_stats['min_distance'] = 0.0
        if extreme_neg_stats['min_distance'] == float('inf'):
            extreme_neg_stats['min_distance'] = 0.0
        
        # Convert all defaultdicts to regular dicts for JSON serialization
        diagnostic_stats['topic_distribution'] = {
            k: dict(v) if isinstance(v, defaultdict) else v
            for k, v in diagnostic_stats['topic_distribution'].items()
        }
        
        # Save main dataset
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save detailed statistics
        stats_path = os.path.join(self.output_dir, f'{os.path.splitext(filename)[0]}_statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(diagnostic_stats, f, indent=2)
        
        # Print summary statistics
        print("\nDataset Generation Complete")
        print("---------------------------")
        print(f"Total triplets generated: {diagnostic_stats['overall_stats']['total_triplets']}")
        print(f"Total positives: {diagnostic_stats['overall_stats']['total_positives']}")
        print(f"Total negatives: {diagnostic_stats['overall_stats']['total_negatives']}")
        
        # Print negative statistics
        hard_neg_stats = diagnostic_stats['negative_sampling_stats']['hard_negatives']
        if hard_neg_stats['same_broad_different_medium'] + hard_neg_stats['different_broad'] > 0:
            print(f"\nHard Negatives:")
            print(f"- Same broad topic: {hard_neg_stats['same_broad_different_medium']}")
            print(f"- Different broad topic: {hard_neg_stats['different_broad']}")
            print(f"- Average distance: {hard_neg_stats['avg_distance']:.2f}")
            print(f"- Distance range: {hard_neg_stats['min_distance']:.2f} to {hard_neg_stats['max_distance']:.2f}")
        
        extreme_neg_stats = diagnostic_stats['negative_sampling_stats']['extreme_negatives']
        if extreme_neg_stats['avg_distance'] > 0:
            print(f"\nExtreme Negatives:")
            print(f"- Unique broad topics: {extreme_neg_stats['unique_broad_topics']}")
            print(f"- Average distance: {extreme_neg_stats['avg_distance']:.2f}")
            print(f"- Distance range: {extreme_neg_stats['min_distance']:.2f} to {extreme_neg_stats['max_distance']:.2f}")
        
        print(f"\nOutput saved to: {output_path}")
        print(f"Detailed statistics saved to: {stats_path}")
        
def main():
    # Configure input/output paths
    input_file = "data/ProTrix_Nomic.csv"  # Change this to your input path
    output_dir = "output"  # Change this to your desired output directory
    
    # Initialize and run the generator
    generator = ContrastiveDatasetGenerator(input_file, output_dir)
    dataset = generator.generate_dataset()
    generator.save_dataset(dataset)

if __name__ == "__main__":
    main()
