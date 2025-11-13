# Generation Caching System

## 🎯 Overview

The generation caching system **automatically saves and reuses model completions** based on high-level parameters, not exact prompt content. This significantly speeds up experiments and reduces VLLM server usage.

## ✅ What Gets Cached

- **Full texts** (prompt + completion)
- **Completions only** (just the generated text)
- **Metadata** (model, temperature, timestamp, subject type, etc.)

## 🔑 Cache Key Design (High-Level Parameters)

**The cache is keyed by high-level parameters, NOT by actual prompt content:**

- **Model** (e.g., `gemma-3-12b`)
- **Prompt format** (e.g., `benign`, `50-50`)
- **Subject type** (e.g., `all`, `math`, `nonmath`, `math_and_nonmath`)
- **Number of samples** (e.g., 200)
- **Temperature** and **max_tokens**

### What This Means

✅ **Smart reuse**: Different questions with the same prompt format reuse the cache
✅ **Subject separation**: Different subjects can have separate caches (e.g., math-only vs all subjects)
✅ **Sample count tracking**: Different sample counts don't interfere
❌ **Intentional regeneration**: Changing prompt format requires regeneration (as expected)

**Example**: If you cache 200 generations with the "50-50" prompt on all subjects, then later run the same experiment with *different* 200 questions but same prompt/model/subject/count, it will reuse the cache. This is the intended behavior!

## 📁 Cache Location

Cached generations are stored in:
```
experiments/{model_short_name}/generations/{prompt_name}/{subject_type}/
```

Example:
```
experiments/gemma-3-12b/generations/50-50/all/n200_t1.0_maxtok512_full_texts.npy
experiments/gemma-3-12b/generations/50-50/all/n200_t1.0_maxtok512_completions.npy
experiments/gemma-3-12b/generations/50-50/all/n200_t1.0_maxtok512_metadata.json
experiments/gemma-3-12b/generations/benign/math_and_nonmath/n400_t1.0_maxtok512_full_texts.npy
...
```

The cache filename encodes:
- **n{num_samples}**: Number of samples (e.g., n200)
- **t{temperature}**: Sampling temperature (e.g., t1.0)
- **maxtok{max_tokens}**: Maximum tokens to generate (e.g., maxtok512)

## 🚀 Benefits

### Before Caching:
```bash
# Layer 10 sweep
python cache_activations.py --prompt "50-50" --layer 10  # Generates 200 completions
# Layer 11 sweep  
python cache_activations.py --prompt "50-50" --layer 11  # Regenerates SAME 200 completions
# Layer 12 sweep
python cache_activations.py --prompt "50-50" --layer 12  # Regenerates SAME 200 completions
```

**Total time**: ~3x generation time + 3x activation extraction

### After Caching:
```bash
# Layer 10 sweep
python cache_activations.py --prompt "50-50" --layer 10  # Generates 200 completions (first time)
# Layer 11 sweep
python cache_activations.py --prompt "50-50" --layer 11  # ✓ Loads cached completions!
# Layer 12 sweep
python cache_activations.py --prompt "50-50" --layer 12  # ✓ Loads cached completions!
```

**Total time**: 1x generation time + 3x activation extraction

## 📊 Usage

### Automatic (Default)

Caching is **enabled by default**. Just run your commands normally:

```bash
# First run - generates and caches
python cache_activations.py --prompt "50-50" --layer 10

# Second run - uses cached generations (even with different questions!)
python cache_activations.py --prompt "50-50" --layer 11

# Layer sweep - only generates once!
python sweep_layers.py --prompt "50-50" --layers 10 11 12 13 14
```

### Subject Types

Different subject types create separate caches:

- **`all`**: All MMLU subjects (used by `cache_activations.py`, `identify_reliable_questions.py`)
- **`math_and_nonmath`**: Mixed math and non-math subjects (used by `math_vs_nonmath_experiment.py`)
- **`math`**, **`nonmath`**: Individual subject types (if needed)

### Cache Reuse Rules

Cache is reused when:
✅ Same prompt format (e.g., both use "50-50")
✅ Same model
✅ Same subject type (e.g., both use "all")
✅ Same number of samples
✅ Same temperature and max_tokens

Cache is NOT reused when:
❌ Different prompt format ("benign" vs "50-50")
❌ Different model
❌ Different subject type ("all" vs "math")
❌ Different number of samples (200 vs 400)
❌ Different temperature or max_tokens

### Disable Caching

To force regeneration (if needed), delete the cache:

```bash
# Delete all cached generations for a specific prompt and subject
rm -rf experiments/gemma-3-12b/generations/50-50/all/

# Delete all cached generations for a prompt
rm -rf experiments/gemma-3-12b/generations/50-50/

# Delete all cached generations
rm -rf experiments/gemma-3-12b/generations/
```

## 🔍 Cache Hits vs Misses

When running a script, you'll see:

**Cache HIT:**
```
============================================================
Checking for cached generations...
  Model: gemma-3-12b
  Prompt: 50-50
  Subject: all
  Samples: 200
  Cache path: experiments/gemma-3-12b/generations/50-50/all/n200_t1.0_maxtok512
============================================================

✓ Found cached generations!
  Loaded 200 cached generations
  Cached on: 2024-11-13 15:30:45
  Model: google/gemma-3-12b-it
  Subject: all
============================================================
```

**Cache MISS:**
```
============================================================
Checking for cached generations...
  Model: gemma-3-12b
  Prompt: 50-50
  Subject: all
  Samples: 200
  Cache path: experiments/gemma-3-12b/generations/50-50/all/n200_t1.0_maxtok512
============================================================
  No cache found - will generate and save

============================================================
Distributing 200 prompts across 8 VLLM servers
...
[Generation happens]
...
  💾 Saving generations to cache: experiments/gemma-3-12b/generations/50-50/all/n200_t1.0_maxtok512
  ✓ Cached 200 generations
```

## 🗂️ Cache Management

### View Cached Generations

```bash
# List all cached generation sets
find experiments/gemma-3-12b/generations -name "*_metadata.json" -exec cat {} \;

# Check cache size
du -sh experiments/gemma-3-12b/generations/

# List caches by prompt
ls -lh experiments/gemma-3-12b/generations/50-50/
ls -lh experiments/gemma-3-12b/generations/benign/
```

### Clear Old Caches

```bash
# Remove caches older than 7 days
find experiments/gemma-3-12b/generations -name "*_metadata.json" -mtime +7 -delete
find experiments/gemma-3-12b/generations -name "*_full_texts.npy" -mtime +7 -delete
find experiments/gemma-3-12b/generations -name "*_completions.npy" -mtime +7 -delete
```

## ⚠️ Important Notes

1. **High-level caching**: Cache reuse is based on prompt FORMAT, not prompt CONTENT. This means different questions with the same format reuse the cache.
2. **Subject separation**: Different subject types (`all`, `math`, `math_and_nonmath`) have separate caches
3. **Sample count matters**: 200 samples and 400 samples create different caches
4. **Disk space**: Cached generations take ~10-100MB per 200 samples depending on completion length
5. **Model changes**: Different models have separate cache directories

## 🔧 Technical Details

**Cache path generation:**
```python
# Path based on high-level parameters:
cache_dir = f"{model_short_name}/generations/{prompt_name}/{subject_type}/"
cache_file = f"n{num_samples}_t{temperature}_maxtok{max_tokens}"
```

**Storage format:**
- Full texts: NumPy array (dtype=object) for efficient storage
- Completions: NumPy array (dtype=object)
- Metadata: JSON file with generation parameters

## 🎓 Example Workflow

```bash
# 1. First layer sweep - generates completions
python sweep_layers.py --prompt "50-50" --layers 10 11 12 13 14 --num-examples 200
# Generation happens once, then reused for layers 11-14

# 2. Re-run for different token position - reuses same completions!
python cache_activations.py --prompt "50-50" --layer 13 --position first
# ✓ Loads cached generations

# 3. Try different layer - still reuses!
python cache_activations.py --prompt "50-50" --layer 20 --position last
# ✓ Loads cached generations

# 4. Run with different questions but same parameters - STILL REUSES!
python cache_activations.py --prompt "50-50" --layer 10
# ✓ Loads cached generations (even if questions are different)

# 5. Change prompt - new cache created
python cache_activations.py --prompt "benign" --layer 10
# Generates new completions (different prompt format)

# 6. Different subject type - new cache created
python math_vs_nonmath_experiment.py  # Uses subject_type="math_and_nonmath"
# Generates new completions (different subject type)
```

## 💡 Pro Tips

1. **Run one full sweep first**: This populates the cache for all subsequent experiments
2. **Keep your cache**: Don't delete it unless you need to regenerate with different questions
3. **Monitor disk space**: Each cache uses ~50-100MB, manageable for most setups
4. **Subject types matter**: Be aware of which subject type your experiment uses
5. **Sample count flexibility**: If you need 100 samples but only have 200 cached, you'll regenerate (no partial reuse yet)

