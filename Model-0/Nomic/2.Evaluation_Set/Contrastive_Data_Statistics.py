import json
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Any
import sys
from pathlib import Path

class ContrastiveDatasetAnalyzer:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        
    def analyze_topic_distribution(self) -> Dict[str, Dict[str, int]]:
        """Analyze the distribution of broad and medium topics."""
        topic_stats = {
            'anchor_topics': defaultdict(int),
            'medium_topics': defaultdict(int),
            'topic_pairs': defaultdict(int)
        }
        
        for item in self.data:
            broad_topic = item['anchor']['broad_topic']
            medium_topic = item['anchor']['medium_topic']
            topic_stats['anchor_topics'][broad_topic] += 1
            topic_stats['medium_topics'][medium_topic] += 1
            topic_stats['topic_pairs'][(broad_topic, medium_topic)] += 1
            
        return {k: dict(v) for k, v in topic_stats.items()}
    
    def analyze_distance_statistics(self) -> Dict[str, Any]:
        """Analyze the distribution of distances for positives and negatives."""
        stats = {
            'positive_distances': [],
            'negative_distances': [],
            'distance_ranges': defaultdict(int)
        }
        
        for item in self.data:
            # Primary positive distance
            stats['positive_distances'].append(item['primary_positive']['distance'])
            
            # Additional positives distances
            for pos in item['additional_positives']:
                stats['positive_distances'].append(pos['distance'])
                
            # Negative distances
            for neg in item['negatives']:
                distance = neg['distance']
                stats['negative_distances'].append(distance)
                
                # Categorize distances into ranges
                if distance <= 2.5:
                    stats['distance_ranges']['0-2.5'] += 1
                elif distance <= 5.0:
                    stats['distance_ranges']['2.5-5.0'] += 1
                elif distance <= 10.0:
                    stats['distance_ranges']['5.0-10.0'] += 1
                else:
                    stats['distance_ranges']['10.0+'] += 1
        
        # Calculate statistics
        if stats['positive_distances']:
            stats['positive_distance_stats'] = {
                'min': min(stats['positive_distances']),
                'max': max(stats['positive_distances']),
                'mean': np.mean(stats['positive_distances']),
                'median': np.median(stats['positive_distances']),
                'std': np.std(stats['positive_distances'])
            }
        
        if stats['negative_distances']:
            stats['negative_distance_stats'] = {
                'min': min(stats['negative_distances']),
                'max': max(stats['negative_distances']),
                'mean': np.mean(stats['negative_distances']),
                'median': np.median(stats['negative_distances']),
                'std': np.std(stats['negative_distances'])
            }
        
        return stats
    
    def analyze_pair_counts(self) -> Dict[str, Any]:
        """Analyze the distribution of positive and negative pairs."""
        counts = {
            'total_examples': len(self.data),
            'positives_per_anchor': [],
            'negatives_per_anchor': [],
            'total_positives': 0,
            'total_negatives': 0
        }
        
        for item in self.data:
            num_positives = 1 + len(item['additional_positives'])  # primary + additional
            num_negatives = len(item['negatives'])
            
            counts['positives_per_anchor'].append(num_positives)
            counts['negatives_per_anchor'].append(num_negatives)
            counts['total_positives'] += num_positives
            counts['total_negatives'] += num_negatives
        
        if counts['positives_per_anchor']:
            counts['avg_positives_per_anchor'] = np.mean(counts['positives_per_anchor'])
        if counts['negatives_per_anchor']:
            counts['avg_negatives_per_anchor'] = np.mean(counts['negatives_per_anchor'])
        
        return counts
    
    def analyze_cross_topic_relationships(self) -> Dict[str, Any]:
        """Analyze relationships between different topics in negative pairs."""
        cross_topic_stats = {
            'broad_topic_transitions': defaultdict(int),
            'medium_topic_transitions': defaultdict(int),
            'broad_topic': {
                'same_topic_negatives': 0,
                'different_topic_negatives': 0
            },
            'medium_topic': {
                'same_topic_negatives': 0,
                'different_topic_negatives': 0
            }
        }
        
        for item in self.data:
            anchor_broad_topic = item['anchor']['broad_topic']
            anchor_medium_topic = item['anchor']['medium_topic']
            
            for neg in item['negatives']:
                neg_broad_topic = neg['broad_topic']
                neg_medium_topic = neg['medium_topic']
                
                # Broad topic transitions
                cross_topic_stats['broad_topic_transitions'][(anchor_broad_topic, neg_broad_topic)] += 1
                if anchor_broad_topic == neg_broad_topic:
                    cross_topic_stats['broad_topic']['same_topic_negatives'] += 1
                else:
                    cross_topic_stats['broad_topic']['different_topic_negatives'] += 1
                
                # Medium topic transitions
                cross_topic_stats['medium_topic_transitions'][(anchor_medium_topic, neg_medium_topic)] += 1
                if anchor_medium_topic == neg_medium_topic:
                    cross_topic_stats['medium_topic']['same_topic_negatives'] += 1
                else:
                    cross_topic_stats['medium_topic']['different_topic_negatives'] += 1
        
        # Convert defaultdicts to regular dicts for JSON serialization
        cross_topic_stats['broad_topic_transitions'] = dict(cross_topic_stats['broad_topic_transitions'])
        cross_topic_stats['medium_topic_transitions'] = dict(cross_topic_stats['medium_topic_transitions'])
        return cross_topic_stats
    
    def generate_full_report(self) -> Dict[str, Any]:
        """Generate a comprehensive statistical report of the dataset."""
        report = {
            'topic_distribution': self.analyze_topic_distribution(),
            'distance_statistics': self.analyze_distance_statistics(),
            'pair_counts': self.analyze_pair_counts(),
            'cross_topic_relationships': self.analyze_cross_topic_relationships()
        }
        return report

def analyze_dataset(input_file: str) -> Dict[str, Any]:
    """Main function to analyze a contrastive dataset file."""
    try:
        # Verify file exists
        if not Path(input_file).exists():
            print(f"Error: File '{input_file}' not found.")
            return {}
            
        # Load the dataset
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or not data:
            print("Error: Invalid or empty dataset format.")
            return {}
            
        # Create analyzer and generate report
        analyzer = ContrastiveDatasetAnalyzer(data)
        report = analyzer.generate_full_report()
        
        return report
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {str(e)}")
        return {}
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")
        return {}

def print_report(report: Dict[str, Any]):
    """Print a formatted version of the analysis report."""
    if not report:
        print("Error: No valid report data available.")
        return
        
    print("\n=== Contrastive Dataset Analysis Report ===\n")
    
    # Basic counts
    if 'pair_counts' in report:
        pair_counts = report['pair_counts']
        print("Basic Statistics:")
        print(f"Total examples: {pair_counts['total_examples']}")
        print(f"Total positive pairs: {pair_counts['total_positives']}")
        print(f"Total negative pairs: {pair_counts['total_negatives']}")
        print(f"Average positives per anchor: {pair_counts.get('avg_positives_per_anchor', 0):.2f}")
        print(f"Average negatives per anchor: {pair_counts.get('avg_negatives_per_anchor', 0):.2f}")
    
    # Distance statistics
    if 'distance_statistics' in report:
        dist_stats = report['distance_statistics']
        print("\nDistance Statistics:")
        
        if 'positive_distance_stats' in dist_stats:
            print("\nPositive Pairs:")
            for key, value in dist_stats['positive_distance_stats'].items():
                print(f"  {key}: {value:.3f}")
        
        if 'negative_distance_stats' in dist_stats:
            print("\nNegative Pairs:")
            for key, value in dist_stats['negative_distance_stats'].items():
                print(f"  {key}: {value:.3f}")
        
        if 'distance_ranges' in dist_stats:
            print("\nNegative Distance Ranges:")
            for range_name, count in dist_stats['distance_ranges'].items():
                print(f"  {range_name}: {count}")
    
    # Topic distribution
    if 'topic_distribution' in report:
        print("\nTopic Distribution:")
        
        if 'medium_topics' in report['topic_distribution']:
            print("\nMedium Topics:")
            # Sort by count in descending order
            sorted_medium_topics = sorted(
                report['topic_distribution']['medium_topics'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for topic, count in sorted_medium_topics:
                percentage = (count / report['pair_counts']['total_examples']) * 100
                print(f"  {topic}: {count} ({percentage:.1f}%)")

        print("\nBroad Topics:")
        if 'anchor_topics' in report['topic_distribution']:
            sorted_broad_topics = sorted(
                report['topic_distribution']['anchor_topics'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for topic, count in sorted_broad_topics:
                percentage = (count / report['pair_counts']['total_examples']) * 100
                print(f"  {topic}: {count} ({percentage:.1f}%)")
    
    # Cross-topic relationships
    if 'cross_topic_relationships' in report:
        cross_topic = report['cross_topic_relationships']
        print("\nCross-topic Statistics:")
        
        # Broad topic relationships
        print("\nBroad Topic Relationships:")
        broad_stats = cross_topic.get('broad_topic', {})
        if broad_stats:
            total_broad = broad_stats.get('same_topic_negatives', 0) + broad_stats.get('different_topic_negatives', 0)
            if total_broad > 0:
                same_broad_pct = (broad_stats.get('same_topic_negatives', 0) / total_broad) * 100
                diff_broad_pct = (broad_stats.get('different_topic_negatives', 0) / total_broad) * 100
                print(f"Same broad topic negatives: {broad_stats.get('same_topic_negatives', 0)} ({same_broad_pct:.1f}%)")
                print(f"Different broad topic negatives: {broad_stats.get('different_topic_negatives', 0)} ({diff_broad_pct:.1f}%)")
        
        # Medium topic relationships
        print("\nMedium Topic Relationships:")
        medium_stats = cross_topic.get('medium_topic', {})
        if medium_stats:
            total_medium = medium_stats.get('same_topic_negatives', 0) + medium_stats.get('different_topic_negatives', 0)
            if total_medium > 0:
                same_medium_pct = (medium_stats.get('same_topic_negatives', 0) / total_medium) * 100
                diff_medium_pct = (medium_stats.get('different_topic_negatives', 0) / total_medium) * 100
                print(f"Same medium topic negatives: {medium_stats.get('same_topic_negatives', 0)} ({same_medium_pct:.1f}%)")
                print(f"Different medium topic negatives: {medium_stats.get('different_topic_negatives', 0)} ({diff_medium_pct:.1f}%)")

def main():
    # Define input directory and file
    base_dir = Path('input')
    input_file = base_dir / 'contrastive_dataset.json'
    
    # Create input directory if it doesn't exist
    base_dir.mkdir(exist_ok=True)
    
    # Check if file exists
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        print("Please place the dataset file in the 'input' directory.")
        sys.exit(1)
        
    report = analyze_dataset(str(input_file))
    print_report(report)
    
    print(f"\nNote: Analysis completed for file: {input_file}")

if __name__ == "__main__":
    main()