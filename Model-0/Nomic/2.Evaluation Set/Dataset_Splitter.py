import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict

class ContrastiveDatasetSplitter:
    def __init__(self, data: List[Dict], train_ratio: float = 0.7, test_ratio: float = 0.15):
        self.data = data
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = 1 - train_ratio - test_ratio
        self.rng = np.random.default_rng(42)
        
    def get_topic_distribution(self) -> Dict[str, List[int]]:
        topic_indices = defaultdict(list)
        for idx, item in enumerate(self.data):
            medium_topic = item['anchor']['medium_topic']
            topic_indices[medium_topic].append(idx)
        return dict(topic_indices)

    def create_oversampled_example(self, example: Dict, oversample_id: int) -> Dict:
        """Create a new dictionary for oversampled example with a unique identifier."""
        new_example = json.loads(json.dumps(example))  # Deep copy
        new_example['oversample_id'] = oversample_id
        # print(f"Created oversampled example {oversample_id} from original ID {example['anchor']['id']}")
        return new_example

    def _get_unique_examples_count(self, dataset: List[Dict]) -> Tuple[int, int]:
        """
        Count unique and oversampled examples in a dataset.
        
        Returns:
            Tuple of (unique_count, oversampled_count)
        """
        seen_ids = set()
        total_count = 0
        
        for item in dataset:
            example_id = item['anchor']['id']
            if example_id not in seen_ids:
                seen_ids.add(example_id)
            total_count += 1
            
        return len(seen_ids), total_count - len(seen_ids)

    def _get_topic_stats(self, dataset: List[Dict], include_oversampled: bool = True) -> Dict[str, Dict[str, int]]:
        """Calculate topic distribution statistics with more detailed tracking."""
        stats = {
            'total_counts': defaultdict(int),
            'unique_counts': defaultdict(int)
        }
        seen_examples = set()  # Track unique IDs per topic
        
        for item in dataset:
            example_id = item['anchor']['id']
            topic = item['anchor']['medium_topic']
            
            # Count unique examples per topic
            if example_id not in seen_examples:
                seen_examples.add(example_id)
                stats['unique_counts'][topic] += 1
            
            # Count total examples (including oversampled)
            if include_oversampled or example_id not in seen_examples:
                stats['total_counts'][topic] += 1
        
        return {
            'total_counts': dict(stats['total_counts']),
            'unique_counts': dict(stats['unique_counts']),
            'oversampled_counts': {
                topic: stats['total_counts'][topic] - stats['unique_counts'][topic]
                for topic in set(stats['total_counts']) | set(stats['unique_counts'])
            }
        }

    def simple_three_way_split(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Perform a simple three-way split of the dataset."""
        topic_indices = self.get_topic_distribution()
        train_indices = []
        test_indices = []
        val_indices = []
        
        for topic, indices in topic_indices.items():
            n_samples = len(indices)
            n_train = int(n_samples * self.train_ratio)
            n_test = int(n_samples * self.test_ratio)
            
            shuffled_indices = self.rng.permutation(indices)
            train_indices.extend(shuffled_indices[:n_train])
            test_indices.extend(shuffled_indices[n_train:n_train + n_test])
            val_indices.extend(shuffled_indices[n_train + n_test:])
        
        train_data = [self.data[i] for i in train_indices]
        test_data = [self.data[i] for i in test_indices]
        val_data = [self.data[i] for i in val_indices]
        
        return train_data, test_data, val_data
    
    def oversample_three_way_split(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Perform a three-way split with oversampling for minority topics."""
        topic_indices = self.get_topic_distribution()
        topic_sizes = {k: len(v) for k, v in topic_indices.items()}
        
        # Calculate median topic size
        median_size = np.median(list(topic_sizes.values()))
        target_min_size = int(median_size * 0.5)  # Set target to 50% of median size
        
        train_data = []
        test_data = []
        val_data = []
        oversample_counter = 0
        
        for topic, indices in topic_indices.items():
            n_samples = len(indices)
            n_train = int(n_samples * self.train_ratio)
            n_test = int(n_samples * self.test_ratio)
            
            shuffled_indices = self.rng.permutation(indices)
            
            # Split indices
            orig_train = shuffled_indices[:n_train]
            test_indices = shuffled_indices[n_train:n_train + n_test]
            val_indices = shuffled_indices[n_train + n_test:]
            
            # Add test and validation examples directly
            test_data.extend(self.data[i] for i in test_indices)
            val_data.extend(self.data[i] for i in val_indices)
            
            # Handle training data with potential oversampling
            if n_samples < target_min_size:
                # Add original samples first
                train_data.extend(self.data[i] for i in orig_train)
                
                # Calculate oversampling factor
                oversample_factor = max(1, int(np.ceil(target_min_size / n_samples))) - 1
                
                if oversample_factor > 0:
                    # Add oversampled copies
                    for _ in range(oversample_factor):
                        for idx in orig_train:
                            train_data.append(
                                self.create_oversampled_example(
                                    self.data[idx],
                                    oversample_counter
                                )
                            )
                            oversample_counter += 1
                            
                    # If we still need more samples to reach target
                    remaining_needed = target_min_size - (n_samples * (oversample_factor + 1))
                    if remaining_needed > 0:
                        extra_samples = self.rng.choice(orig_train, size=int(remaining_needed), replace=False)
                        for idx in extra_samples:
                            train_data.append(
                                self.create_oversampled_example(
                                    self.data[idx],
                                    oversample_counter
                                )
                            )
                            oversample_counter += 1
            else:
                # Add original samples without oversampling
                train_data.extend(self.data[i] for i in orig_train)
        
        return train_data, test_data, val_data

    def save_splits(self, output_dir: Path, train_data: List[Dict], 
                   val_data: List[Dict], test_data: List[Dict], suffix: str = "") -> Dict:
        """Save the splits and generate detailed statistics."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        for split_name, split_data in [
            ('train', train_data),
            ('test', test_data),
            ('val', val_data)
        ]:
            file_path = output_dir / f'{split_name}{suffix}.json'
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        # Get unique and oversampled counts
        train_unique, train_oversampled = self._get_unique_examples_count(train_data)
        
        # Generate detailed statistics
        stats = {
            'dataset_sizes': {
                'original_total': len(self.data),
                'train': {
                    'total': len(train_data),
                    'unique': train_unique,
                    'oversampled': train_oversampled
                },
                'test': len(test_data),
                'val': len(val_data)
            },
            'topic_distribution': {
                'original': self._get_topic_stats(self.data),
                'train': self._get_topic_stats(train_data),
                'test': self._get_topic_stats(test_data),
                'val': self._get_topic_stats(val_data)
            }
        }
        
        # Calculate oversampling statistics
        stats['oversampling_metrics'] = {
            'train_oversampling_factor': (
                len(train_data) / train_unique
                if train_unique > 0 else 1.0
            ),
            'total_examples_after_oversampling': (
                len(train_data) + len(test_data) + len(val_data)
            )
        }
        
        # Save statistics
        stats_file = output_dir / f'split_statistics{suffix}.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Print summary statistics
        print(f"\nDataset Split Statistics ({suffix}):")
        print("-" * 40)
        print(f"Original dataset size: {stats['dataset_sizes']['original_total']}")
        print("\nSplit sizes:")
        print(f"Train: {stats['dataset_sizes']['train']['total']} "
              f"(unique: {stats['dataset_sizes']['train']['unique']}, "
              f"oversampled: {stats['dataset_sizes']['train']['oversampled']})")
        print(f"Test: {stats['dataset_sizes']['test']}")
        print(f"Val: {stats['dataset_sizes']['val']}")
        print(f"\nOversampling factor: {stats['oversampling_metrics']['train_oversampling_factor']:.2f}x")
        
        return stats

def main():
    base_dir = Path('output')
    simple_dir = base_dir / 'simple_split'
    oversample_dir = base_dir / 'oversample_split'
    
    input_file = Path('input') / 'contrastive_dataset.json'
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        splitter = ContrastiveDatasetSplitter(data, train_ratio=0.7, test_ratio=0.15)
        
        # Generate splits
        print("Generating simple three-way split...")
        train_simple, test_simple, val_simple = splitter.simple_three_way_split()
        simple_stats = splitter.save_splits(simple_dir, train_simple, val_simple, test_simple, "_simple")
        
        print("Generating oversampled three-way split...")
        train_over, test_over, val_over = splitter.oversample_three_way_split()
        over_stats = splitter.save_splits(oversample_dir, train_over, val_over, test_over, "_oversampled")
        
        # Print summary
        print("\nSplit Generation Complete!")
        
        print("\nSimple Split Statistics:")
        print(f"Training set: {len(train_simple)} examples ({len(train_simple)/len(data)*100:.1f}%)")
        print(f"Test set: {len(test_simple)} examples ({len(test_simple)/len(data)*100:.1f}%)")
        print(f"Validation set: {len(val_simple)} examples ({len(val_simple)/len(data)*100:.1f}%)")
        
        print("\nOversampled Split Statistics:")
        # Get counts of unique examples and oversampled examples
        unique_train = len({item['anchor']['id'] for item in train_over})
        total_train = len(train_over)
        oversampled = total_train - unique_train
        print(f"Training set: {total_train} examples (unique: {unique_train}, oversampled: {oversampled})")
        print(f"Test set: {len(test_over)} examples")
        print(f"Validation set: {len(val_over)} examples")
        
        print("\nOutput directories:")
        print(f"Simple split: {simple_dir}")
        print(f"Oversampled split: {oversample_dir}")
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main()

