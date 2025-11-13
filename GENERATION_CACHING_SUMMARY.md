# Generation Caching Update Summary

## ✅ What's New

The generation caching system has been enhanced to save cached generations in **three formats** for maximum flexibility:

1. **NumPy** (`.npy`) - Efficient binary format for Python code
2. **JSON** (`.json`) - Structured data for programmatic inspection
3. **Markdown** (`.md`) - Beautiful formatting for human reading

## 🎯 Quick Start

### For Existing Cached Generations

Decode your existing numpy files to JSON and Markdown:

```bash
# Decode with a reasonable limit for readability
python decode_generations.py --model gemma-3-12b --limit 10
```

### For Future Generations

No action needed! All new generations will automatically be saved in all three formats.

## 📁 What You Get

After decoding (or generating new data), you'll have:

```
experiments/gemma-3-12b/generations/50-50_filtered/all/
├── n1374_t1.0_maxtok100_completions.npy    # NumPy (for Python)
├── n1374_t1.0_maxtok100_completions.json   # JSON (structured)
├── n1374_t1.0_maxtok100_completions.md     # Markdown (beautiful!)
├── n1374_t1.0_maxtok100_full_texts.npy
├── n1374_t1.0_maxtok100_full_texts.json
├── n1374_t1.0_maxtok100_full_texts.md
└── n1374_t1.0_maxtok100_metadata.json
```

## 🔧 Tools Provided

### 1. Decode Script (`decode_generations.py`)

Converts existing numpy files to JSON and Markdown:

```bash
# Basic usage
python decode_generations.py --model gemma-3-12b --limit 10

# See all options
python decode_generations.py --help
```

### 2. Updated Caching Function (`lib/generation.py`)

The `save_generations_to_cache()` function now automatically saves all three formats:

```python
from lib.generation import save_generations_to_cache

# This will create .npy, .json, and .md files
save_generations_to_cache(cache_path, full_texts, completions, metadata)
```

## 📖 Example: Reading Generations

### Markdown (Best for Humans)

Open in your favorite editor or viewer:

```bash
cat experiments/gemma-3-12b/generations/50-50_filtered/all/n1374_t1.0_maxtok100_completions.md
```

You'll see beautifully formatted output:

```markdown
# Completions Only

**Model:** google/gemma-2-27b-it
**Generated:** 2025-11-13 02:52:15
**Count:** 10 (showing first 10 of 1374)

---

## Completion 1

The correct answer is C. The passage explicitly states...

---

## Completion 2

The passage explicitly states that students bring...

---
```

### JSON (For Scripts)

Use `jq` or any JSON tool:

```bash
jq '.[0]' experiments/gemma-3-12b/generations/50-50_filtered/all/n1374_t1.0_maxtok100_completions.json
```

### NumPy (For Python Code)

Python code continues to work as before:

```python
from lib.generation import load_generations_from_cache

# Loads from .npy files (efficient)
full_texts, completions, metadata = load_generations_from_cache(cache_path)
```

## 💡 Best Practices

1. **Use `--limit 10` when decoding large files** - Creates manageable files for inspection
2. **Markdown for reading** - Most pleasant format for human browsing
3. **JSON for scripts** - Great for quick analysis or tools
4. **NumPy for Python** - Most efficient for loading into code
5. **Disk space aware** - JSON/Markdown are 2-5x larger than NumPy; use limits if needed

## 🔄 Backwards Compatibility

✅ All existing code continues to work without changes!

- `load_generations_from_cache()` still loads from NumPy files
- JSON and Markdown files are optional add-ons
- No breaking changes to any APIs

## 📚 Documentation

- **[DECODE_GENERATIONS.md](DECODE_GENERATIONS.md)** - Full decode script documentation
- **[GENERATION_CACHING.md](GENERATION_CACHING.md)** - Original caching system docs

## 🎉 Benefits

- **Better debugging**: Easily inspect what the model generated
- **No more numpy headaches**: No need to load Python just to see what's in a file
- **Beautiful formatting**: Markdown makes browsing generations a pleasure
- **Multiple formats**: Choose the right tool for the job
- **No downsides**: Existing workflows unchanged

## 🚀 Try It Now

```bash
# Decode your existing generations
python decode_generations.py --model gemma-3-12b --limit 10

# View in markdown
cat experiments/gemma-3-12b/generations/50-50_filtered/all/n1374_t1.0_maxtok100_completions.md

# Future generations will automatically have all three formats!
```

---

**Questions?** See [DECODE_GENERATIONS.md](DECODE_GENERATIONS.md) for detailed documentation.


