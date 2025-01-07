import os
import json
import math
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
from tqdm import tqdm

class ContrastiveDatasetGenerator:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        self.hard_negative_threshold = 5.0
        
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
        except:
            return table_str
    
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
                'sentence_context': para['sentence_context']
            })
            
        return additional_positives
    
    def sample_negatives(self, anchor: pd.Series, topic_to_exclude: str, 
                        paragraphs: pd.DataFrame, num_hard: int = 5, 
                        num_extreme: int = 5) -> List[Dict[str, Any]]:
        """Sample negatives with relaxed constraints and better range utilization."""
        negatives = []
        
        # Get paragraphs from other topics
        other_topic_paragraphs = paragraphs[
            paragraphs['Nomic Topic: broad'] != topic_to_exclude
        ].copy()
        
        # Calculate distances once
        other_topic_paragraphs['distance'] = other_topic_paragraphs.apply(
            lambda x: self.get_distance(anchor, x), axis=1
        )
        
        # Get hard negatives (≤5.0 distance)
        hard_candidates = other_topic_paragraphs[other_topic_paragraphs['distance'] <= self.hard_negative_threshold]
        hard_negatives = []
        used_topics = set()

        # First attempt: Different topics
        for _, group in hard_candidates.groupby('Nomic Topic: broad'):
            if len(hard_negatives) >= num_hard:
                break
            
            if group['Nomic Topic: broad'].iloc[0] not in used_topics:
                sample = group.sample(n=1).iloc[0]
                hard_negatives.append({
                    'negative_id': int(sample['id']),
                    'distance': float(sample['distance']),
                    'broad_topic': sample['Nomic Topic: broad'],
                    'medium_topic': sample['Nomic Topic: medium'],
                    'sentence_context': sample['sentence_context']
                })
                used_topics.add(sample['Nomic Topic: broad'])

        # Second attempt: Relax topic constraints if needed
        if len(hard_negatives) < num_hard:
            remaining_hard = num_hard - len(hard_negatives)
            remaining_candidates = hard_candidates[
                ~hard_candidates['id'].isin([n['negative_id'] for n in hard_negatives])
            ]
            
            if len(remaining_candidates) > 0:  # Only sample if we have candidates
                samples = remaining_candidates.sample(n=min(remaining_hard, len(remaining_candidates)))
                for _, sample in samples.iterrows():
                    hard_negatives.append({
                        'negative_id': int(sample['id']),
                        'distance': float(sample['distance']),
                        'broad_topic': sample['Nomic Topic: broad'],
                        'medium_topic': sample['Nomic Topic: medium'],
                        'sentence_context': sample['sentence_context']
                    })

        # Get extreme negatives (from farthest)
        excluded_ids = [n['negative_id'] for n in hard_negatives]
        extreme_candidates = other_topic_paragraphs[
            ~other_topic_paragraphs['id'].isin(excluded_ids)
        ].sort_values('distance', ascending=False)
        
        extreme_negatives = []
        used_topics = set()

        # First attempt: Different topics from farthest
        for _, sample in extreme_candidates.iterrows():
            if len(extreme_negatives) >= num_extreme:
                break
            
            if (sample['distance'] > self.hard_negative_threshold and 
                sample['Nomic Topic: broad'] not in used_topics):
                extreme_negatives.append({
                    'negative_id': int(sample['id']),
                    'distance': float(sample['distance']),
                    'broad_topic': sample['Nomic Topic: broad'],
                    'medium_topic': sample['Nomic Topic: medium'],
                    'sentence_context': sample['sentence_context']
                })
                used_topics.add(sample['Nomic Topic: broad'])

        # Second attempt: Relax topic constraints if needed
        if len(extreme_negatives) < num_extreme:
            remaining_extreme = num_extreme - len(extreme_negatives)
            remaining_candidates = extreme_candidates[
                (extreme_candidates['distance'] > self.hard_negative_threshold) &
                ~extreme_candidates['id'].isin([n['negative_id'] for n in extreme_negatives])
            ]
            
            if len(remaining_candidates) > 0:  # Only take samples if we have candidates
                samples = remaining_candidates.head(remaining_extreme)
                for _, sample in samples.iterrows():
                    extreme_negatives.append({
                        'negative_id': int(sample['id']),
                        'distance': float(sample['distance']),
                        'broad_topic': sample['Nomic Topic: broad'],
                        'medium_topic': sample['Nomic Topic: medium'],
                        'sentence_context': sample['sentence_context']
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
            
            # Sample negatives
            negatives = self.sample_negatives(table, table_broad_topic, paragraphs)
            
            # Create triplet
            triplet = {
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
                    'sentence_context': matching_paragraph['sentence_context']
                },
                'additional_positives': additional_positives,
                'negatives': negatives
            }
            
            contrastive_data.append(triplet)
        
        return contrastive_data
    
    def save_dataset(self, data: List[Dict[str, Any]], filename: str = 'contrastive_dataset.json'):
        """Save dataset and detailed statistics, but print only summary stats."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Collect detailed diagnostic statistics
        diagnostic_stats = {
            'overall_stats': {
                'total_triplets': len(data),
                'total_negatives': sum(len(d['negatives']) for d in data),
                'triplets_with_insufficient_negatives': sum(1 for d in data if len(d['negatives']) < 10)
            },
            'distance_distributions': {},
            'topic_distributions': {},
            'problematic_cases': []
        }
        
        # Analyze negative samples
        negative_distances = []
        topics_used = set()
        
        for triplet in data:
            if len(triplet['negatives']) < 10:
                diagnostic_stats['problematic_cases'].append({
                    'anchor_id': triplet['anchor']['id'],
                    'anchor_topic': triplet['anchor']['broad_topic'],
                    'negatives_found': len(triplet['negatives'])
                })
            
            for neg in triplet['negatives']:
                negative_distances.append(neg['distance'])
                topics_used.add(neg['broad_topic'])
        
        # Calculate distance statistics
        if negative_distances:
            diagnostic_stats['distance_distributions'] = {
                'min': min(negative_distances),
                'max': max(negative_distances),
                'mean': sum(negative_distances) / len(negative_distances),
                'by_range': {
                    '0-2.5': sum(1 for d in negative_distances if d <= 2.5),
                    '2.5-5.0': sum(1 for d in negative_distances if 2.5 < d <= 5.0),
                    '5.0-10.0': sum(1 for d in negative_distances if 5.0 < d <= 10.0),
                    '10.0+': sum(1 for d in negative_distances if d > 10.0)
                }
            }
        
        # Save main dataset and diagnostic statistics
        output_path = os.path.join(self.output_dir, filename)
        stats_path = os.path.join(self.output_dir, 'detailed_statistics.json')
        
        # Save main dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save detailed statistics
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(diagnostic_stats, f, indent=2)
        
        # Print only summary statistics
        total_triplets = len(data)
        total_negatives = sum(len(d['negatives']) for d in data)
        total_positives = sum(len(d['additional_positives']) for d in data) + total_triplets
        
        print("\nDataset Generation Complete")
        print("---------------------------")
        print(f"Total triplets generated: {total_triplets}")
        print(f"Total positives (primary + additional): {total_positives}")
        print(f"Total negatives: {total_negatives}")
        print(f"Average negatives per triplet: {total_negatives/total_triplets:.2f}")
        print(f"Average positives per triplet: {total_positives/total_triplets:.2f}")
        print(f"Output saved to: {output_path}")
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