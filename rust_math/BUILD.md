# Building rust_math

## One-time setup on VPS

### 1. Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Install maturin (Python build tool for Rust)
```bash
pip install maturin
```

### 3. Build the Rust extension
```bash
cd ~/polyMM/rust_math
maturin develop --release
```

### 4. Test it works
```bash
cd ~/polyMM/rust_math
python test_rust_math.py
```

You should see accuracy tests pass and speedup numbers (10-50x faster).

## Using in your bot

In `util.py`, change the imports:

```python
# Option 1: Use Rust (fast)
from rust_math import bs_binary_call, bs_binary_call_delta, bs_binary_call_implied_vol_closed

# Option 2: Use Python (fallback)
# from util import bs_binary_call, bs_binary_call_delta, bs_binary_call_implied_vol_closed
```

Or add a try/except to auto-fallback:

```python
try:
    from rust_math import bs_binary_call, bs_binary_call_delta, bs_binary_call_implied_vol_closed
    print("[INIT] Using Rust math functions")
except ImportError:
    from util import bs_binary_call, bs_binary_call_delta, bs_binary_call_implied_vol_closed
    print("[INIT] Rust not available, using Python math functions")
```

## Rebuilding after changes

If you modify the Rust code:
```bash
cd ~/polyMM/rust_math
maturin develop --release
```

## Troubleshooting

**"rust_math not found"**: Make sure you ran `maturin develop --release` in the rust_math directory.

**Build errors**: Make sure Rust is installed (`rustc --version`) and you sourced the cargo env.
