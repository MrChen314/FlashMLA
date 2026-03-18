# Head128 MLA Backward Two-Kernel Flow

This diagram splits the current fused backward path into two kernels:

- `Kernel 1` recomputes `P / S / dP / dS`, accumulates `dQ`, and stores `S` and `dS` to global scratch.
- `Kernel 2` reloads `S` and `dS` from global scratch and computes `dKV` directly, without recomputing softmax-related intermediates.

```mermaid
flowchart LR
    classDef gm fill:#e8f2ff,stroke:#3b82f6,stroke-width:1px,color:#0f172a;
    classDef load fill:#efe7ff,stroke:#8b5cf6,stroke-width:1px,color:#111827;
    classDef smem fill:#fff4d6,stroke:#f59e0b,stroke-width:1px,color:#111827;
    classDef mma fill:#e8f7e8,stroke:#22c55e,stroke-width:1px,color:#111827;
    classDef scratch fill:#fff1e6,stroke:#f97316,stroke-width:1px,color:#111827;
    classDef atomic fill:#ffe8e8,stroke:#ef4444,stroke-width:1px,color:#111827;

    subgraph GM["Global Memory"]
        KGM["K[k]<br/>Key Block k"]
        QGM["Q<br/>Query Matrix"]
        DOGM["dO<br/>Output Gradient"]
        VGM["V[k]<br/>Value Block k"]
        LSEGM["LSE<br/>Log-Sum-Exp"]
        DELTAGM["Delta<br/>Precomputed"]
        IDXGM["TopK Indices / Valid Mask"]
        DQGM["dQ_global<br/>Query Gradient"]
        DKVGM["dKV_global<br/>Packed dK + dV"]
    end

    subgraph K1["Kernel 1: Compute dQ and Save S / dS"]
        direction LR

        subgraph K1LOAD["TMA Copy / Gather Load"]
            K1LK["Load K[k]"]
            K1LQ["Load Q"]
            K1LDO["Load dO"]
            K1LV["Load V[k]"]
        end

        subgraph K1SMEM["Shared Memory / TMEM"]
            K1SK["sK[k]"]
            K1SQ["sQ"]
            K1SDO["sdO"]
            K1SV["sV[k]"]
            K1TP["P[k] in TMEM"]
            K1TDP["dP_mid[k] in TMEM"]
            K1SS["sS[k]<br/>Softmax Output"]
            K1SDS["sdS[k]<br/>Softmax Gradient"]
        end

        subgraph K1MMA["MMA Compute"]
            K1P["Step 1: Recompute Attention<br/>P[k] = Q @ K[k]^T"]
            K1SOFT["Step 2: Scale + Softmax<br/>S[k] = exp2(P * scale - LSE)"]
            K1DP["Step 3: dP<br/>dP_mid[k] = dO @ V[k]^T"]
            K1DS["Step 4: dS<br/>dS[k] = S[k] * (dP_mid[k] - Delta) * scale"]
            K1DQ["Step 5: dQ accumulation<br/>dQ += dS[k] @ K[k]"]
        end
    end

    subgraph SCRATCH["Global Scratch Between Kernels"]
        SGM["S_scratch[k]<br/>Persist S for each KV block"]
        DSGM["dS_scratch[k]<br/>Persist dS for each KV block"]
    end

    subgraph K2["Kernel 2: Load S / dS and Compute dKV"]
        direction LR

        subgraph K2LOAD["TMA Copy / Gather Load"]
            K2LK["Load K[k]"]
            K2LQ["Load Q"]
            K2LDO["Load dO"]
            K2LV["Load V[k]"]
            K2LS["Load S_scratch[k]"]
            K2LDS["Load dS_scratch[k]"]
        end

        subgraph K2SMEM["Shared Memory / TMEM"]
            K2SK["sK[k]"]
            K2SQ["sQ"]
            K2SDO["sdO"]
            K2SV["sV[k]"]
            K2SS["sS[k]"]
            K2SDS["sdS[k]"]
        end

        subgraph K2MMA["MMA Compute"]
            K2NOTE["No softmax recompute<br/>Reuse stored S and dS directly"]
            K2DV["Step 1: dV accumulation<br/>dV[k] += S[k]^T @ dO"]
            K2DKN["Step 2: dK_nope accumulation<br/>dK_nope[k] += dS[k]^T @ Q_nope"]
            K2DKR["Step 3: dK_rope accumulation<br/>dK_rope[k] += dS[k]^T @ Q_rope"]
            K2PACK["Step 4: Pack / merge as dKV tile"]
        end

        subgraph K2ATOMIC["Atomic Operations"]
            K2ATOM["atomic_add(dKV_global, dKV[k])"]
        end
    end

    KGM --> K1LK
    QGM --> K1LQ
    DOGM --> K1LDO
    VGM --> K1LV
    IDXGM --> K1LK
    IDXGM --> K1SOFT
    LSEGM --> K1SOFT
    DELTAGM --> K1DS

    K1LK --> K1SK
    K1LQ --> K1SQ
    K1LDO --> K1SDO
    K1LV --> K1SV

    K1SK --> K1P
    K1SQ --> K1P
    K1P --> K1TP
    K1TP --> K1SOFT
    K1SOFT --> K1SS

    K1SDO --> K1DP
    K1SV --> K1DP
    K1DP --> K1TDP
    K1TDP --> K1DS
    K1SS --> K1DS
    K1DS --> K1SDS

    K1SDS --> K1DQ
    K1SK --> K1DQ
    K1DQ --> DQGM

    K1SS --> SGM
    K1SDS --> DSGM

    KGM --> K2LK
    QGM --> K2LQ
    DOGM --> K2LDO
    VGM --> K2LV
    IDXGM --> K2LK
    SGM --> K2LS
    DSGM --> K2LDS

    K2LK --> K2SK
    K2LQ --> K2SQ
    K2LDO --> K2SDO
    K2LV --> K2SV
    K2LS --> K2SS
    K2LDS --> K2SDS

    K2NOTE --> K2DV
    K2NOTE --> K2DKN
    K2NOTE --> K2DKR

    K2SS --> K2DV
    K2SDO --> K2DV
    K2SDS --> K2DKN
    K2SQ --> K2DKN
    K2SDS --> K2DKR
    K2SQ --> K2DKR

    K2DV --> K2PACK
    K2DKN --> K2PACK
    K2DKR --> K2PACK
    K2PACK --> K2ATOM
    K2ATOM --> DKVGM

    class KGM,QGM,DOGM,VGM,LSEGM,DELTAGM,IDXGM,DQGM,DKVGM gm;
    class K1LK,K1LQ,K1LDO,K1LV,K2LK,K2LQ,K2LDO,K2LV,K2LS,K2LDS load;
    class K1SK,K1SQ,K1SDO,K1SV,K1TP,K1TDP,K1SS,K1SDS,K2SK,K2SQ,K2SDO,K2SV,K2SS,K2SDS smem;
    class K1P,K1SOFT,K1DP,K1DS,K1DQ,K2NOTE,K2DV,K2DKN,K2DKR,K2PACK mma;
    class SGM,DSGM scratch;
    class K2ATOM atomic;
```

Notes:

- `Kernel 1` keeps the original backward front half: `P -> S -> dP -> dS`, then finishes `dQ`.
- `Kernel 2` removes all softmax reconstruction work and consumes `S` / `dS` as reusable intermediates.
- The split is aimed at reducing the resource pressure of the current fused `dQ + dKV` kernel.
