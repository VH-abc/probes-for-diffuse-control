# Decoding Cached Generation Files

## Overview

The generation caching system now supports three formats:
- **NumPy** (`.npy`): Efficient storage for loading into Python
- **JSON** (`.json`): Human-readable structured data
- **Markdown** (`.md`): Beautiful formatting for easy browsing

## 🎯 Quick Start

### Decode Existing NumPy Files

To convert existing cached generation files from NumPy to JSON and Markdown:

```bash
# Decode all cached generations (limit to first 10 items per file for readability)
python decode_generations.py --limit 10

# Decode for a specific model with more items
python decode_generations.py --model gemma-3-12b --limit 50

# Decode all items (warning: large files!)
python decode_generations.py --model gemma-3-12b

# Preview what would be decoded (dry run)
python decode_generations.py --model gemma-3-12b --dry-run

# Force overwrite existing JSON/Markdown files
python decode_generations.py --model gemma-3-12b --force
```

### Future Generations (Automatic Multi-Format Caching)

Starting now, all new generations will be automatically cached in all three formats:
- **NumPy files** (`.npy`): Efficient storage for loading into Python
- **JSON files** (`.json`): Structured data for programmatic inspection
- **Markdown files** (`.md`): Beautiful formatting for human reading

The `save_generations_to_cache()` function in `lib/generation.py` has been updated to automatically save all three formats.

## 📁 File Structure

After decoding, you'll have:

```
experiments/
└── gemma-3-12b/
    └── generations/
        ├── benign/
        │   └── all/
        │       ├── n20000_t1.0_maxtok100_completions.npy    # Original NumPy (efficient)
        │       ├── n20000_t1.0_maxtok100_completions.json   # Structured JSON
        │       ├── n20000_t1.0_maxtok100_completions.md     # NEW: Beautiful Markdown
        │       ├── n20000_t1.0_maxtok100_full_texts.npy
        │       ├── n20000_t1.0_maxtok100_full_texts.json
        │       ├── n20000_t1.0_maxtok100_full_texts.md      # NEW: Beautiful Markdown
        │       └── n20000_t1.0_maxtok100_metadata.json
        └── 50/
            └── 50_filtered/
                └── all/
                    ├── n1374_t1.0_maxtok100_completions.npy
                    ├── n1374_t1.0_maxtok100_completions.json
                    ├── n1374_t1.0_maxtok100_completions.md
                    ├── n1374_t1.0_maxtok100_full_texts.npy
                    ├── n1374_t1.0_maxtok100_full_texts.json
                    ├── n1374_t1.0_maxtok100_full_texts.md
                    └── n1374_t1.0_maxtok100_metadata.json
```

## 🔧 Usage Examples

### View Completions in Markdown

```bash
# Open markdown file in your favorite editor/viewer (beautifully formatted!)
cat experiments/gemma-3-12b/generations/50-50_filtered/all/n1374_t1.0_maxtok100_completions.md

# View in VS Code, Cursor, or any markdown viewer
code experiments/gemma-3-12b/generations/50-50_filtered/all/n1374_t1.0_maxtok100_completions.md
```

### View Completions in JSON

```bash
# View first few completions in JSON format
head -n 50 experiments/gemma-3-12b/generations/benign/all/n20000_t1.0_maxtok100_completions.json | jq
```

### Search for Specific Text

```bash
# Search in markdown files (most readable results)
grep -i "correct answer" experiments/gemma-3-12b/generations/50-50_filtered/all/*.md

# Search in JSON files (structured results)
grep -i "correct answer" experiments/gemma-3-12b/generations/50-50_filtered/all/*.json
```

### Count Items

```bash
# Count number of items in JSON file
jq 'length' experiments/gemma-3-12b/generations/50-50_filtered/all/n1374_t1.0_maxtok100_completions.json

# Count generations in markdown file
grep -c "^## Completion" experiments/gemma-3-12b/generations/50-50_filtered/all/n1374_t1.0_maxtok100_completions.md
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

1. **Use `--limit` for large files**: For files with thousands of generations, use `--limit 10` or `--limit 50` to create manageable files for human inspection

2. **Markdown for reading**: Markdown files are the most pleasant for human reading - nicely formatted with headers and separators

3. **JSON for programmatic use**: JSON files are great for quick scripts or tools that need structured data

4. **NumPy for Python code**: Python code should continue to use the NumPy files for efficiency

5. **Disk space**: Markdown and JSON files are typically 2-5x larger than NumPy files. Use `--limit` if disk space is a concern

6. **Automatic caching**: New generations will automatically create all three formats. No need to run the decode script for new generations!

## 🔄 Backwards Compatibility

The loading functions (`load_generations_from_cache()`) continue to work exactly as before:
- They load from NumPy files (`.npy`)
- JSON and Markdown files are optional and only for human inspection
- Existing code requires no changes

## 📊 Example Output

When you run the decode script, you'll see:

```
============================================================
Decoding Cached Generation Files
============================================================
Model: gemma-3-12b
Limiting to first 10 items per file
============================================================

Found 4 generation files

Processing: experiments/gemma-3-12b/generations/50-50_filtered/all/n1374_t1.0_maxtok100_completions.npy
    (limiting to first 10 of 1374 items)
  ✓ Decoded 10 items to JSON and Markdown

Processing: experiments/gemma-3-12b/generations/50-50_filtered/all/n1374_t1.0_maxtok100_full_texts.npy
    (limiting to first 10 of 1374 items)
  ✓ Decoded 10 items to JSON and Markdown

============================================================
Summary:
  Successfully decoded: 4
  Skipped (already exists): 0
  Total processed: 4/4
============================================================
```

## 📖 Example Markdown Output

The markdown files are beautifully formatted:

```markdown
# Completions Only

**Model:** google/gemma-2-27b-it
**Generated:** 2025-11-13 02:52:15
**Count:** 10 (showing first 10 of 1374)

---

## Completion 1

The correct answer is C. The passage explicitly states that not knowing 
who Alan Greenspan is can lead to "potentially embarrassing conversations."

---

## Completion 2

The passage explicitly states that students bring their lunches to school...

---
```

## 🐛 Troubleshooting

**Q: The script doesn't find my generation files**  
A: Make sure you're in the project root directory and the files are in `experiments/MODEL_NAME/generations/`

**Q: Markdown/JSON files are too large**  
A: Use `--limit 10` to only save the first 10 items for inspection

**Q: Can I delete the JSON and Markdown files?**  
A: Yes! The NumPy files are the source of truth. JSON and Markdown files are just for human inspection and can be regenerated anytime.

**Q: Do I need to update my code?**  
A: No! All existing code continues to work. The JSON and Markdown files are optional additions for human readability.

**Q: Which format should I use for viewing?**  
A: Markdown (`.md`) files are the most pleasant for reading - they're nicely formatted with headers and separators. Use JSON for programmatic access.

