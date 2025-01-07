import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class ContrastiveDatasetSplitter:
    def __init__(self, data: List[Dict], train_ratio: float = 0.8):
        self.data = data
        self.train_ratio = train_ratio
        self.rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
    def get_topic_distribution(self) -> Dict[str, List[int]]:
        """Get distribution of examples by medium topic."""
        topic_indices = defaultdict(list)
        for idx, item in enumerate(self.data):
            medium_topic = item['anchor']['medium_topic']
            topic_indices[medium_topic].append(idx)
        return dict(topic_indices)

    def simple_split(self) -> Tuple[List[Dict], List[Dict]]:
        """Perform stratified split without oversampling."""
        topic_indices = self.get_topic_distribution()
        train_indices = []
        eval_indices = []
        
        for topic, indices in topic_indices.items():
            n_samples = len(indices)
            n_train = int(n_samples * self.train_ratio)
            
            # Shuffle indices
            shuffled_indices = self.rng.permutation(indices)
            
            # Split indices
            train_indices.extend(shuffled_indices[:n_train])
            eval_indices.extend(shuffled_indices[n_train:])
        
        # Create dataset splits
        train_data = [self.data[i] for i in train_indices]
        eval_data = [self.data[i] for i in eval_indices]
        
        return train_data, eval_data
    
    def oversample_split(self) -> Tuple[List[Dict], List[Dict]]:
        """Perform stratified split with oversampling for small topics."""
        topic_indices = self.get_topic_distribution()
        topic_sizes = {k: len(v) for k, v in topic_indices.items()}
        
        # Calculate median topic size
        median_size = np.median(list(topic_sizes.values()))
        min_eval_samples = 5  # Minimum samples in evaluation set
        
        train_indices = []
        eval_indices = []
        
        for topic, indices in topic_indices.items():
            n_samples = len(indices)
            
            if n_samples < median_size * 0.2:  # Small topic
                # Ensure minimum evaluation samples
                n_eval = min(min_eval_samples, max(1, int(n_samples * 0.3)))
                n_train = n_samples - n_eval
                
                # Oversample training set
                oversample_factor = max(1, int(median_size * 0.2 / n_train))
                
                # Shuffle indices
                shuffled_indices = self.rng.permutation(indices)
                
                # Split and oversample
                train_subset = list(shuffled_indices[:n_train])
                eval_subset = list(shuffled_indices[n_train:])
                
                # Add oversampled training indices
                train_indices.extend(train_subset * oversample_factor)
                eval_indices.extend(eval_subset)
            else:
                # Regular split for larger topics
                n_train = int(n_samples * self.train_ratio)
                shuffled_indices = self.rng.permutation(indices)
                train_indices.extend(shuffled_indices[:n_train])
                eval_indices.extend(shuffled_indices[n_train:])
        
        # Create dataset splits
        train_data = [self.data[i] for i in train_indices]
        eval_data = [self.data[i] for i in eval_indices]
        
        return train_data, eval_data

    def save_splits(self, output_dir: Path, train_data: List[Dict], 
                   eval_data: List[Dict], suffix: str = ""):
        """Save the splits to JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training set
        train_file = output_dir / f'train{suffix}.json'
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
            
        # Save evaluation set
        eval_file = output_dir / f'eval{suffix}.json'
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        # Generate and save statistics
        stats = {
            'total_examples': len(self.data),
            'train_examples': len(train_data),
            'eval_examples': len(eval_data),
            'train_ratio': len(train_data) / len(self.data),
            'topic_distribution': {
                'original': self._get_topic_stats(self.data),
                'train': self._get_topic_stats(train_data),
                'eval': self._get_topic_stats(eval_data)
            }
        }
        
        stats_file = output_dir / f'split_statistics{suffix}.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            
        return stats
    
    def _get_topic_stats(self, dataset: List[Dict]) -> Dict[str, int]:
        """Calculate topic distribution statistics."""
        topic_counts = defaultdict(int)
        for item in dataset:
            topic_counts[item['anchor']['medium_topic']] += 1
        return dict(topic_counts)

def main():
    # Setup directories
    base_dir = Path('output')
    simple_dir = base_dir / 'simple_split'
    oversample_dir = base_dir / 'oversample_split'
    
    # Load dataset
    input_file = Path('input') / 'contrastive_dataset.json'
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Create splitter
        splitter = ContrastiveDatasetSplitter(data)
        
        # Generate and save simple split
        print("Generating simple split...")
        train_simple, eval_simple = splitter.simple_split()
        simple_stats = splitter.save_splits(simple_dir, train_simple, eval_simple, "_simple")
        
        # Generate and save oversampled split
        print("Generating oversampled split...")
        train_over, eval_over = splitter.oversample_split()
        over_stats = splitter.save_splits(oversample_dir, train_over, eval_over, "_oversampled")
        
        # Print summary
        print("\nSplit Generation Complete!")
        print("\nSimple Split Statistics:")
        print(f"Training set: {len(train_simple)} examples")
        print(f"Evaluation set: {len(eval_simple)} examples")
        
        print("\nOversampled Split Statistics:")
        print(f"Training set: {len(train_over)} examples")
        print(f"Evaluation set: {len(eval_over)} examples")
        
        print("\nOutput directories:")
        print(f"Simple split: {simple_dir}")
        print(f"Oversampled split: {oversample_dir}")
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main()