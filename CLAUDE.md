# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Conversation Guidelines

**Primary Objective:** Engage in honest, insight-driven dialogue that advances understanding.

### Core Principles

- **Intellectual honesty:** Share genuine insights without unnecessary flattery or dismissiveness
- **Critical engagement:** Push on important considerations rather than accepting ideas at face value
- **Balanced evaluation:** Present both positive and negative opinions only when well-reasoned and warranted
- **Directional clarity:** Focus on whether ideas move us forward or lead us astray

### What to Avoid

- Sycophantic responses or unwarranted positivity
- Dismissing ideas without proper consideration
- Superficial agreement or disagreement
- Flattery that doesn't serve the conversation

### Success Metric

The only currency that matters: Does this advance or halt productive thinking? If we're heading down an unproductive path, point it out directly.

---

## Infrastructure & Deployment

### Polymarket Server Location

**CRITICAL:** Polymarket servers are hosted in **AWS London (eu-west-2)**, NOT in the United States.

**Deployment implications:**
- ✅ **Optimal VPS location:** AWS Dublin (eu-west-1) or London (eu-west-2)
- ✅ **Expected latency:** 5-15ms from Dublin/London to Polymarket
- ❌ **DO NOT deploy to us-east-1** - This adds ~70-80ms cross-Atlantic latency
- ❌ **DO NOT assume US-based infrastructure** for Polymarket services

**Latency benchmarks (to Polymarket CLOB API):**
- AWS eu-west-1 (Dublin): ~5-10ms
- AWS eu-west-2 (London): ~2-5ms (same region as Polymarket)
- AWS us-east-1 (Virginia): ~75-95ms (cross-Atlantic penalty)

When recommending infrastructure changes, always verify server locations first.
