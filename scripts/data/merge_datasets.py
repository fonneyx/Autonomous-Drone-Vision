#!/usr/bin/env python3
"""
Unified YOLO Dataset Merger
Combines multiple YOLO datasets into a single Master Dataset with:
- Unified class namespace (preventing duplicate class names)
- Automatic class ID re-indexing in label files
- Collision-free filename management
"""

import os
import sys
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Fix Windows console encoding for emoji/special characters
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')


class YOLODatasetMerger:
    def __init__(self, source_datasets: List[str], output_dir: str):
        """
        Initialize the merger with source datasets and output directory.
        
        Args:
            source_datasets: List of paths to YOLO dataset directories
            output_dir: Path to Master_Dataset output directory
        """
        self.source_datasets = [Path(d) for d in source_datasets]
        self.output_dir = Path(output_dir)
        self.unified_classes: Dict[str, int] = {}
        self.class_mapping: Dict[str, Dict[int, int]] = defaultdict(dict)
        
    def parse_dataset_yaml(self, dataset_path: Path) -> Tuple[Dict[int, str], str]:
        """
        Parse data.yaml from a YOLO dataset to extract class names.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Tuple of (class_dict {id: name}, split_name)
        """
        yaml_path = dataset_path / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found in {dataset_path}")
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        classes = data.get('names', {})
        if isinstance(classes, list):
            classes = {i: name for i, name in enumerate(classes)}
        
        return classes, dataset_path.name
    
    def build_unified_class_namespace(self) -> None:
        """
        Parse all datasets and build a unified class namespace.
        Assigns new indices to ensure no duplicates across all datasets.
        """
        print("\n[INFO] Building unified class namespace...")
        unified_index = 0
        
        for dataset_path in self.source_datasets:
            classes, dataset_name = self.parse_dataset_yaml(dataset_path)
            print(f"\n  Dataset: {dataset_name}")
            print(f"    Classes: {classes}")
            
            for old_id, class_name in classes.items():
                if class_name not in self.unified_classes:
                    self.unified_classes[class_name] = unified_index
                    print(f"      -> '{class_name}' -> Global ID {unified_index}")
                    unified_index += 1
                else:
                    print(f"      -> '{class_name}' -> Global ID {self.unified_classes[class_name]} (already exists)")
                
                self.class_mapping[dataset_name][old_id] = self.unified_classes[class_name]
        
        print(f"\n[OK] Unified namespace: {self.unified_classes}")
    
    def create_master_yaml(self) -> None:
        """
        Create the master data.yaml with unified class namespace.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        master_yaml = {
            'path': str(self.output_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.unified_classes),
            # Ultralytics expects {class_id: class_name}
            'names': {idx: name for name, idx in self.unified_classes.items()}
        }
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(master_yaml, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n[OK] Created master data.yaml at {yaml_path}")
    
    def reindex_labels(self, label_content: str, dataset_name: str) -> str:
        """
        Re-index class IDs in label file content using class mapping.
        
        Args:
            label_content: Content of .txt label file
            dataset_name: Name of source dataset
            
        Returns:
            Re-indexed label content
        """
        lines = label_content.strip().split('\n')
        reindexed_lines = []
        
        for line in lines:
            if not line.strip():
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            old_class_id = int(parts[0])
            new_class_id = self.class_mapping[dataset_name][old_class_id]
            
            reindexed_lines.append(f"{new_class_id} {' '.join(parts[1:])}")
        
        return '\n'.join(reindexed_lines) + '\n' if reindexed_lines else ''
    
    def copy_dataset_split(self, split_name: str) -> None:
        """
        Copy images and labels for a specific split (train, val, test).
        Prevents filename collisions by prepending dataset folder name.
        
        Args:
            split_name: One of 'train', 'val', 'test'
        """
        output_images_dir = self.output_dir / split_name / 'images'
        output_labels_dir = self.output_dir / split_name / 'labels'
        
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)
        
        print(f"\n[INFO] Processing '{split_name}' split...")
        
        for dataset_path in self.source_datasets:
            source_split = split_name
            # Roboflow exports name the validation split 'valid'
            if split_name == 'val' and not (dataset_path / 'val').exists():
                source_split = 'valid'
            dataset_split_images = dataset_path / source_split / 'images'
            dataset_split_labels = dataset_path / source_split / 'labels'
            dataset_name = dataset_path.name
            
            if not dataset_split_images.exists():
                print(f"  [WARN] {dataset_name}/{split_name}/images not found, skipping")
                continue
            
            file_count = 0
            for image_file in dataset_split_images.glob('*'):
                if not image_file.is_file():
                    continue
                
                # Prepend dataset name to prevent collisions
                new_image_name = f"{dataset_name}_{image_file.name}"
                output_image_path = output_images_dir / new_image_name
                shutil.copy2(image_file, output_image_path)
                
                # Process corresponding label file
                label_file = dataset_split_labels / f"{image_file.stem}.txt"
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        label_content = f.read()
                    
                    reindexed_content = self.reindex_labels(label_content, dataset_name)
                    
                    new_label_name = f"{dataset_name}_{image_file.stem}.txt"
                    output_label_path = output_labels_dir / new_label_name
                    with open(output_label_path, 'w') as f:
                        f.write(reindexed_content)
                
                file_count += 1
            
            print(f"  [OK] {dataset_name}: Copied {file_count} files to {split_name}")
    
    def merge(self) -> None:
        """
        Execute the complete merge process.
        """
        print("\n" + "="*70)
        print("[START] YOLO DATASET MERGER - STARTING")
        print("="*70)
        
        # Validate input datasets
        for dataset_path in self.source_datasets:
            if not dataset_path.exists():
                raise ValueError(f"Dataset not found: {dataset_path}")
            print(f"[OK] Found dataset: {dataset_path.name}")
        
        # Build unified class namespace
        self.build_unified_class_namespace()
        
        # Create master data.yaml
        self.create_master_yaml()
        
        # Copy and process each split
        for split in ['train', 'val', 'test']:
            try:
                self.copy_dataset_split(split)
            except Exception as e:
                print(f"  [WARN] Error processing '{split}': {e}")
        
        print("\n" + "="*70)
        print("[COMPLETE] MERGE COMPLETE!")
        print("="*70)
        print(f"Master Dataset Location: {self.output_dir}")
        print(f"Total Classes: {len(self.unified_classes)}")
        print(f"Class Mapping: {dict(self.unified_classes)}")
        print("="*70 + "\n")


def main():
    """
    Main execution point for dataset merger.
    """
    # Configuration
    # Datasets live next to the repository folder
    parent_dir = Path(__file__).resolve().parents[3]  # scripts/data/ -> repo -> folder with the datasets
    output_dir = parent_dir / "Master_Dataset"
    
    source_datasets = [
        parent_dir / "Drone.v1i.yolov11",
        parent_dir / "phone.v1i.yolov11",
        parent_dir / "vehicles.v2-release.yolov11"
    ]
    
    # Initialize and run merger
    merger = YOLODatasetMerger(
        source_datasets=[str(d) for d in source_datasets],
        output_dir=str(output_dir)
    )
    
    merger.merge()


if __name__ == "__main__":
    main()
