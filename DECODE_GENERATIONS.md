# Decoding Cached Generation Files

## Overview

The generation caching system now supports both **NumPy** (efficient storage) and **JSON** (human-readable) formats.

## 🎯 Quick Start

### Decode Existing NumPy Files

To convert existing cached generation files from NumPy to JSON:

```bash
# Decode all cached generations (limit to first 100 items per file for readability)
python decode_generations.py --limit 100

# Decode for a specific model
python decode_generations.py --model gemma-3-12b --limit 100

# Decode all items (warning: large files!)
python decode_generations.py --model gemma-3-12b

# Preview what would be decoded (dry run)
python decode_generations.py --model gemma-3-12b --dry-run

# Force overwrite existing JSON files
python decode_generations.py --model gemma-3-12b --force
```

### Future Generations (Automatic JSON Caching)

Starting now, all new generations will be automatically cached in both formats:
- **NumPy files** (`.npy`): Efficient storage for loading into Python
- **JSON files** (`.json`): Human-readable format for inspection

The `save_generations_to_cache()` function in `lib/generation.py` has been updated to automatically save both formats.

## 📁 File Structure

After decoding, you'll have:

```
experiments/
└── gemma-3-12b/
    └── generations/
        ├── benign/
        │   └── all/
        │       ├── n20000_t1.0_maxtok100_completions.npy    # Original NumPy (efficient)
        │       ├── n20000_t1.0_maxtok100_completions.json   # NEW: Human-readable JSON
        │       ├── n20000_t1.0_maxtok100_full_texts.npy
        │       ├── n20000_t1.0_maxtok100_full_texts.json
        │       └── n20000_t1.0_maxtok100_metadata.json
        └── 50/
            └── 50_filtered/
                └── all/
                    ├── n1374_t1.0_maxtok100_completions.npy
                    ├── n1374_t1.0_maxtok100_completions.json
                    ├── n1374_t1.0_maxtok100_full_texts.npy
                    ├── n1374_t1.0_maxtok100_full_texts.json
                    └── n1374_t1.0_maxtok100_metadata.json
```

## 🔧 Usage Examples

### View Completions Only

```bash
# View first few completions in human-readable format
head -n 50 experiments/gemma-3-12b/generations/benign/all/n20000_t1.0_maxtok100_completions.json | jq
```

### Search for Specific Text

```bash
# Search for completions containing specific text
grep -i "correct answer" experiments/gemma-3-12b/generations/50/50_filtered/all/n1374_t1.0_maxtok100_completions.json
```

### Count Items

```bash
# Count number of items in JSON file
jq 'length' experiments/gemma-3-12b/generations/50/50_filtered/all/n1374_t1.0_maxtok100_completions.json
```

## ⚙️ Command Line Options

```
decode_generations.py [OPTIONS]

Options:
  --base-dir DIR       Base directory to search (default: experiments)
  --model MODEL        Filter to specific model (e.g., gemma-3-12b)
  --limit N            Limit items saved per file (useful for large files)
  --dry-run            Show what would be done without writing files
  --force              Overwrite existing JSON files
  -h, --help           Show help message
```

## 💡 Best Practices

1. **Use `--limit` for large files**: For files with thousands of generations, use `--limit 100` or similar to create manageable JSON files for human inspection

2. **JSON is for humans**: The JSON files are meant for quick inspection and debugging. Python code should continue to use the NumPy files for efficiency

3. **Disk space**: JSON files are typically 2-3x larger than NumPy files. Use `--limit` if disk space is a concern

4. **Automatic caching**: New generations will automatically create both NumPy and JSON files. No need to run the decode script for new generations!

## 🔄 Backwards Compatibility

The loading functions (`load_generations_from_cache()`) continue to work exactly as before:
- They load from NumPy files (`.npy`)
- JSON files are optional and only for human inspection
- Existing code requires no changes

## 📊 Example Output

When you run the decode script, you'll see:

```
============================================================
Decoding Cached Generation Files
============================================================
Model: gemma-3-12b
Limiting to first 100 items per file
============================================================

Found 4 generation files

Processing: experiments/gemma-3-12b/generations/50/50_filtered/all/n1374_t1.0_maxtok100_completions.npy
    (limiting to first 100 items)
  ✓ Decoded 100 items to experiments/gemma-3-12b/generations/50/50_filtered/all/n1374_t1.0_maxtok100_completions.json

Processing: experiments/gemma-3-12b/generations/50/50_filtered/all/n1374_t1.0_maxtok100_full_texts.npy
    (limiting to first 100 items)
  ✓ Decoded 100 items to experiments/gemma-3-12b/generations/50/50_filtered/all/n1374_t1.0_maxtok100_full_texts.json

============================================================
Summary:
  Successfully decoded: 4
  Skipped (already exists): 0
  Total processed: 4/4
============================================================
```

## 🐛 Troubleshooting

**Q: The script doesn't find my generation files**  
A: Make sure you're in the project root directory and the files are in `experiments/MODEL_NAME/generations/`

**Q: JSON files are too large**  
A: Use `--limit 100` to only save the first 100 items for inspection

**Q: Can I delete the JSON files?**  
A: Yes! The NumPy files are the source of truth. JSON files are just for human inspection and can be regenerated anytime.

**Q: Do I need to update my code?**  
A: No! All existing code continues to work. The JSON files are an optional addition for human readability.

