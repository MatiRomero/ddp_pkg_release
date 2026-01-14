#!/usr/bin/env python3
"""Transform AD dual CSV files to swap coordinates for 1D equivalence.

This script transforms type keys from ((0, 0), (0, y)) to ((0, 0), (y, 0))
to handle coordinate system equivalence in 1D flattened jobs.

Usage:
    python scripts/transform_ad_duals_coords.py input.csv output.csv
    python scripts/transform_ad_duals_coords.py input1.csv output1.csv input2.csv output2.csv ...
    python scripts/transform_ad_duals_coords.py --in-place input.csv [input2.csv ...]
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path
from typing import Iterator


def transform_type_key(type_str: str) -> str:
    """Transform ((0, 0), (0, y)) to ((0, 0), (y, 0)) for 1D equivalence.
    
    Args:
        type_str: String representation of type tuple like "((0, 0), (0, 23))"
        
    Returns:
        Transformed type string like "((0, 0), (23, 0))", or original if not transformable
    """
    try:
        parsed = ast.literal_eval(type_str)
        if isinstance(parsed, tuple) and len(parsed) == 2:
            origin, dest = parsed
            if origin == (0, 0) and isinstance(dest, tuple) and len(dest) == 2:
                if dest[0] == 0 and dest[1] != 0:
                    # Transform ((0, 0), (0, y)) -> ((0, 0), (y, 0))
                    return f"((0, 0), ({dest[1]}, 0))"
                elif dest[0] != 0 and dest[1] == 0:
                    # Already in format ((0, 0), (x, 0)), no change needed
                    return type_str
    except (ValueError, SyntaxError, TypeError):
        # If parsing fails, return original unchanged
        pass
    return type_str  # Return unchanged if we can't transform


def transform_csv(input_path: Path, output_path: Path) -> int:
    """Transform a single CSV file by swapping coordinates in type column.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        
    Returns:
        Number of rows transformed
    """
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        return 0
    
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows_transformed = 0
    
    with input_path.open('r', newline='') as infile, \
         output_path.open('w', newline='') as outfile:
        reader = csv.DictReader(infile)
        
        if 'type' not in reader.fieldnames:
            print(f"Error: CSV file missing 'type' column: {input_path}", file=sys.stderr)
            return 0
        
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        for row in reader:
            original_type = row['type']
            transformed_type = transform_type_key(original_type)
            if transformed_type != original_type:
                rows_transformed += 1
            row['type'] = transformed_type
            writer.writerow(row)
    
    return rows_transformed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'files',
        nargs='+',
        type=Path,
        help='Input CSV file(s). With --in-place, transforms files in place. '
             'Otherwise, must provide pairs: input1 output1 [input2 output2 ...]'
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Transform files in place (overwrite originals). '
             'If set, all arguments are treated as input files.'
    )
    
    args = parser.parse_args()
    
    if args.in_place:
        # All arguments are input files to transform in place
        input_files = args.files
        for input_path in input_files:
            if not input_path.exists():
                print(f"Error: File does not exist: {input_path}", file=sys.stderr)
                return 1
            
            # Create temporary output path
            output_path = input_path.with_suffix(input_path.suffix + '.tmp')
            rows_transformed = transform_csv(input_path, output_path)
            
            if rows_transformed > 0:
                # Replace original with transformed version
                output_path.replace(input_path)
                print(f"Transformed {rows_transformed} rows in {input_path}")
            else:
                # No changes, remove temp file
                output_path.unlink()
                print(f"No transformations needed in {input_path}")
        
        return 0
    
    # Must have pairs of input/output files
    if len(args.files) % 2 != 0:
        print(
            "Error: Must provide pairs of input and output files, "
            "or use --in-place to transform files in place.",
            file=sys.stderr
        )
        return 1
    
    # Process file pairs
    for i in range(0, len(args.files), 2):
        input_path = args.files[i]
        output_path = args.files[i + 1]
        
        rows_transformed = transform_csv(input_path, output_path)
        
        if rows_transformed > 0:
            print(f"Transformed {rows_transformed} rows: {input_path} -> {output_path}")
        else:
            print(f"No transformations needed: {input_path} -> {output_path}")
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

