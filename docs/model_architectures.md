# Model architecture diagrams

This document maps every distinct learned architecture used by the repository.
Experiment names that only change a loss, training schedule, or inference mode
share a diagram. The six classical interpolation/filtering baselines are not
neural models and therefore do not have layer diagrams.

All tensor shapes use `B x C x H x W`. Except for the RGB hybrid, learned models
receive normalized, full-resolution `[Y, Cr, Cb]`; `Cr` and `Cb` have already
been downsampled to 4:2:0 and reconstructed to `H x W` by the configured
interpolator.

For conference posters and slide decks, use the large-type vector exports in
[`docs/figures/model-architectures`](figures/model-architectures/README.md). The
SVG files are best for PowerPoint/Keynote and the PDFs are best for print or
LaTeX workflows.

## Architecture index

| Architecture | Repository models and experiments | Default trainable parameters | Implementation |
| --- | --- | ---: | --- |
| V5 U-Net generator | `v5`, V5.1, `v5_unet_gan`, `v5_prism_mse` | 1,925,667 | [`UNetGenerator`](../chroma/models.py) |
| V5 PatchGAN discriminator | V5 and V5.1 adversarial training | 694,241 | [`Discriminator`](../chroma/models.py) |
| V6 residual refiner | `v6`, `v6_residual` | 593,794 | [`ChromaRefiner`](../chroma/models.py) |
| Configurable residual refiner | `base_v6_retrained`, `no_luma`, `direct_chroma`, `no_block_skips`, depth/width/loss ablations | 593,794 at the base settings | [`AblationChromaRefiner`](../chroma/research_models.py) |
| Chroma SRCNN | `srcnn_chroma` | 19,298 at width 64 | [`SrcnnChromaRefiner`](../chroma/research_models.py) |
| NAF-style refiner | `naf_style_chroma`, `prism_naf` | 249,218 at width 64 and depth 8 | [`NAFStyleChromaRefiner`](../chroma/research_models.py) |
| V7 polar probabilistic refiner | all `v7_*` experiments and mean/safe inference | 595,525 | [`V7PolarChromaRefiner`](../chroma/v7.py) |
| Prism configurable refiner | all `prism_*` experiments | 249,218–1,938,178 in checked-in production configurations | [`PrismRefiner`](../chroma/prism_models.py) |
| Prism conditional discriminator | `prism_gan` training only | 94,977 | [`PrismDiscriminator`](../chroma/prism_models.py) |
| Li2026 neural hybrid | `v6_*` and `prism_residual_*` hybrid runs | selected backbone plus one scalar gain | [`HybridRefiner`](../chroma/li2026_hybrid.py) |
| Hosted TabPFN adapter | per-image `TabPFNRegressor` pilot | hosted model; not inspectable here | [`reconstruct`](../chroma/tabpfn.py) |

`prism_mse`, `prism_edge`, and `prism_forward` use the same Prism residual
architecture as `prism_residual`. Likewise, `prism_polar_forward` shares the
`prism_polar_prob` architecture. Loss edges are shown only where they alter the
trainable graph, as in the adversarial discriminators.

## Shared observation pipeline

```mermaid
flowchart LR
    RGB["Source RGB<br/>B x 3 x H x W"] --> YCC["Convert to YCrCb"]
    YCC --> Y["Full-resolution Y<br/>B x 1 x H x W"]
    YCC --> C["Cr/Cb<br/>B x 2 x H x W"]
    C --> DS["4:2:0 downsample<br/>configured siting + prefilter"]
    DS --> LOW["Low-resolution Cr/Cb<br/>B x 2 x H/2 x W/2"]
    LOW --> UP["Reconstruction filter<br/>usually bilinear"]
    UP --> CUP["Baseline Cr/Cb<br/>B x 2 x H x W"]
    Y --> CAT["Concatenate"]
    CUP --> CAT
    CAT --> X["Model input X = [Y, Cr, Cb]<br/>B x 3 x H x W"]
```

## V5 U-Net generator and PatchGAN

V5 regenerates all three channels. Its encoder has three pooling operations, so
inputs are padded to a multiple of eight for inference. V5.1 uses exactly the
same architecture. The matched `v5_prism_mse` run wraps the generator with an
observed-luma passthrough and does not use the discriminator.

```mermaid
flowchart TB
    X["Input YCrCb<br/>3 x H x W"] --> E1["ConvBlock<br/>3 to 32, H x W"]
    E1 --> P1["MaxPool 2 x 2"]
    P1 --> E2["ConvBlock<br/>32 to 64, H/2 x W/2"]
    E2 --> P2["MaxPool 2 x 2"]
    P2 --> E3["ConvBlock<br/>64 to 128, H/4 x W/4"]
    E3 --> P3["MaxPool 2 x 2"]
    P3 --> E4["ConvBlock bottleneck<br/>128 to 256, H/8 x W/8"]

    E4 --> U3["Transpose conv 2 x 2<br/>256 to 128"]
    E3 -. skip .-> C3["Concatenate: 256 channels"]
    U3 --> C3
    C3 --> D3["ConvBlock<br/>256 to 128"]

    D3 --> U2["Transpose conv 2 x 2<br/>128 to 64"]
    E2 -. skip .-> C2["Concatenate: 128 channels"]
    U2 --> C2
    C2 --> D2["ConvBlock<br/>128 to 64"]

    D2 --> U1["Transpose conv 2 x 2<br/>64 to 32"]
    E1 -. skip .-> C1["Concatenate: 64 channels"]
    U1 --> C1
    C1 --> D1["ConvBlock<br/>64 to 32"]
    D1 --> OUT["1 x 1 conv, 32 to 3<br/>sigmoid"]
    OUT --> YHAT["Generated YCrCb<br/>3 x H x W"]
```

Each `ConvBlock` is `3 x 3 convolution -> ReLU -> 3 x 3 convolution -> ReLU`.

```mermaid
flowchart LR
    REAL["Real target YCrCb<br/>3 x H x W"] --> D
    FAKE["Generated YCrCb<br/>3 x H x W"] --> D
    D["V5 PatchGAN discriminator"] --> C1["4 x 4 conv, stride 2<br/>3 to 32 + LeakyReLU"]
    C1 --> C2["4 x 4 conv, stride 2<br/>32 to 64 + LeakyReLU"]
    C2 --> C3["4 x 4 conv, stride 2<br/>64 to 128 + LeakyReLU"]
    C3 --> C4["4 x 4 conv, stride 2<br/>128 to 256 + LeakyReLU"]
    C4 --> LOGITS["4 x 4 conv, stride 1<br/>256 to 1 patch logits"]
```

## V6 residual chroma refiner

V6 predicts only a two-channel correction. The original luminance bypasses the
network unchanged.

```mermaid
flowchart LR
    X["Input X = [Y, Cr, Cb]<br/>3 x H x W"] --> HEAD["3 x 3 conv<br/>3 to 64"]
    HEAD --> BODY["8 residual blocks<br/>64 channels"]
    BODY --> TAIL["3 x 3 conv<br/>64 to 2"]
    TAIL --> R["Chroma residual R<br/>2 x H x W"]
    X --> C0["Select baseline [Cr, Cb]"]
    C0 --> ADD["Add"]
    R --> ADD
    X --> Y["Select Y"]
    Y --> JOIN["Concatenate"]
    ADD --> JOIN
    JOIN --> OUT["[Y, Cr + Rcr, Cb + Rcb]<br/>3 x H x W"]
```

```mermaid
flowchart LR
    F["Block input<br/>64 channels"] --> C1["3 x 3 conv<br/>64 to 64"]
    C1 --> RELU["ReLU"]
    RELU --> C2["3 x 3 conv<br/>64 to 64"]
    C2 --> ADD["Add"]
    F -. identity skip .-> ADD
    ADD --> O["Block output<br/>64 channels"]
```

## Configurable residual ablation family

The research ablation model generalizes V6 with explicit switches. Loss-only
experiments do not alter this forward graph.

```mermaid
flowchart LR
    X["Input [Y, Cr, Cb]"] --> GUIDE{"Luma guidance?"}
    GUIDE -- "yes" --> THREE["Use 3 channels"]
    GUIDE -- "no" --> TWO["Use [Cr, Cb] only"]
    THREE --> HEAD["3 x 3 conv<br/>input to width"]
    TWO --> HEAD
    HEAD --> BLOCKS["depth configurable blocks<br/>two 3 x 3 convs each"]
    BLOCKS --> BSKIP{"Block residuals?"}
    BSKIP -- "yes" --> RB["Add each block input"]
    BSKIP -- "no" --> PB["Plain stacked conv blocks"]
    RB --> TAIL["3 x 3 conv<br/>width to 2"]
    PB --> TAIL
    TAIL --> GLOBAL{"Global residual?"}
    X --> BASE["Baseline [Cr, Cb]"]
    BASE --> GLOBAL
    GLOBAL -- "yes" --> ADD["baseline + prediction"]
    GLOBAL -- "no" --> SIG["sigmoid prediction"]
    X --> Y["Unchanged Y"]
    ADD --> JOIN["Concatenate with Y"]
    SIG --> JOIN
    Y --> JOIN
```

Checked-in variants change `depth` (4/8/12), `width` (32/64/96), luma input,
per-block skips, or global residual prediction. `loss_mse`, `loss_charbonnier`,
and `loss_l1_edge` keep the base forward graph.

## Chroma SRCNN

```mermaid
flowchart LR
    X["Input: 3 channels with luma<br/>or 2 chroma channels"] --> C1["9 x 9 conv<br/>input to width"]
    C1 --> R1["ReLU"]
    R1 --> C2["1 x 1 conv<br/>width to width/2"]
    C2 --> R2["ReLU"]
    R2 --> C3["5 x 5 conv<br/>width/2 to 2"]
    C3 --> MODE{"Global residual?"}
    X --> C0["Baseline [Cr, Cb]"]
    C0 --> MODE
    MODE -- "yes" --> RES["baseline + prediction"]
    MODE -- "no" --> DIRECT["sigmoid prediction"]
    X --> Y["Unchanged Y"]
    RES --> OUT["Concatenate with Y"]
    DIRECT --> OUT
    Y --> OUT
```

## NAF-style chroma refiner

The outer refiner uses the same chroma-residual contract as V6, but replaces
ReLU residual blocks with nonlinear-activation-free restoration blocks.

```mermaid
flowchart LR
    X["Input: YCrCb or chroma only"] --> HEAD["3 x 3 conv<br/>input to width"]
    HEAD --> BODY["8 NAF-style blocks<br/>default width 64"]
    BODY --> TAIL["3 x 3 conv<br/>width to 2"]
    TAIL --> ADD["Add baseline chroma<br/>or sigmoid for direct mode"]
    X --> C0["Select baseline [Cr, Cb]"]
    C0 --> ADD
    ADD --> OUT["Concatenate with unchanged Y"]
```

```mermaid
flowchart TB
    X["Block input x<br/>C channels"] --> LN1["LayerNorm2d"]
    LN1 --> EXP1["1 x 1 conv<br/>C to 2C"]
    EXP1 --> DW["3 x 3 depthwise conv<br/>2C channels"]
    DW --> G1["SimpleGate<br/>split and multiply, 2C to C"]
    G1 --> CA["Channel attention<br/>global average pool + 1 x 1 conv"]
    G1 --> MUL["Multiply"]
    CA --> MUL
    MUL --> PROJ1["1 x 1 conv<br/>C to C"]
    PROJ1 --> BETA["Scale by learned beta"]
    X --> ADD1["Add"]
    BETA --> ADD1
    ADD1 --> LN2["LayerNorm2d"]
    LN2 --> EXP2["1 x 1 conv<br/>C to 2C"]
    EXP2 --> G2["SimpleGate<br/>2C to C"]
    G2 --> PROJ2["1 x 1 conv<br/>C to C"]
    PROJ2 --> GAMMA["Scale by learned gamma"]
    ADD1 --> ADD2["Add"]
    GAMMA --> ADD2
    ADD2 --> OUT["Block output"]
```

## V7 probabilistic polar refiner

V7 keeps V6's stem and eight-block trunk, but the five-map head predicts a polar
mean correction and two uncertainty parameters. `mean` and `safe` are two
post-processing paths from the same trained architecture.

```mermaid
flowchart TB
    X["Input [Y, Cr, Cb]<br/>3 x H x W"] --> HEAD["3 x 3 conv<br/>3 to 64"]
    HEAD --> BODY["8 V6 residual blocks<br/>64 channels"]
    BODY --> TAIL["3 x 3 conv<br/>64 to 5 raw maps"]

    X --> POLAR0["Baseline chroma to polar<br/>A0, phi0"]
    TAIL --> DA["delta amplitude"]
    TAIL --> PV["2-D phase vector"]
    TAIL --> SA["raw amplitude scale"]
    TAIL --> SK["raw phase concentration"]

    DA --> AMEAN["Amean = clamp(A0 + delta A)<br/>to valid chroma range"]
    POLAR0 --> AMEAN
    PV --> NORM["Normalize vector<br/>delta phi = atan2(sin, cos)"]
    NORM --> PMEAN["phimean = wrap(phi0 + delta phi)"]
    POLAR0 --> PMEAN
    AMEAN --> CART["Polar to Cartesian chroma"]
    PMEAN --> CART
    CART --> MEAN["Mean output<br/>[Y, Crmean, Cbmean]"]
    X --> MEAN

    SA --> SCALE["softplus + epsilon<br/>Laplace scale bA"]
    SK --> KAPPA["softplus + clamp<br/>von Mises kappa"]
    SCALE --> CONF["Amplitude confidence"]
    KAPPA --> CONF2["Phase confidence<br/>also weighted by Amean"]
    POLAR0 --> BLEND["Blend polar correction<br/>back toward A0, phi0"]
    AMEAN --> BLEND
    PMEAN --> BLEND
    CONF --> BLEND
    CONF2 --> BLEND
    BLEND --> SAFE["Safe output<br/>[Y, Crsafe, Cbsafe]"]
    X --> SAFE
```

The deterministic, probabilistic, forward-consistency, safe-inference, and
self-training V7 experiments all instantiate this same five-map network. Their
loss terms, data stage, or selected output differ.

## Prism configurable family

Prism combines three backbone choices, two output representations, optional
known-degradation conditioning, and optional uncertainty heads in one model.

```mermaid
flowchart TB
    X["Input [Y, Cr, Cb]<br/>3 x H x W"] --> HEAD["3 x 3 conv<br/>3 to width"]
    SPEC["Known degradation<br/>x/y siting + 5-filter one-hot"] --> FC["Linear 7 to width"]
    FC -. optional broadcast add .-> HEADOUT["Conditioned features"]
    HEAD --> HEADOUT
    HEADOUT --> CHOICE{"Backbone"}
    CHOICE -- "residual" --> RES["depth V6 residual blocks"]
    CHOICE -- "NAF" --> NAF["depth NAF-style blocks"]
    CHOICE -- "U-Net" --> UNET["3-level U-Net features"]
    RES --> F["Shared feature map"]
    NAF --> F
    UNET --> F

    F --> REP{"Representation"}
    REP -- "Cartesian" --> CT["3 x 3 conv<br/>width to 2 residual maps"]
    CT --> CADD["baseline chroma + residual"]
    REP -- "polar" --> PT["3 x 3 conv<br/>width to delta A and phase vector"]
    PT --> PDEC["smooth positive amplitude<br/>plus circular phase decode"]
    CADD --> IMAGE["Mean image with unchanged Y"]
    PDEC --> IMAGE

    F -. optional detached path .-> UH["3 x 3 uncertainty head<br/>width to 2"]
    UH --> U["Laplace scale + von Mises kappa"]
    IMAGE --> PRED["PrismPrediction"]
    U --> PRED
    X --> PRED
    PRED --> SAFE["Optional safe image<br/>confidence blend toward baseline"]
```

### Prism U-Net backbone

```mermaid
flowchart TB
    F0["Head features<br/>width, H x W"] --> P1["Average pool 2 x 2"]
    P1 --> D1["ConvBlock<br/>width to 2w"]
    D1 --> P2["Average pool 2 x 2"]
    P2 --> D2["ConvBlock<br/>2w to 4w"]
    D2 --> P3["Average pool 2 x 2"]
    P3 --> D3["ConvBlock<br/>4w to 8w"]
    D3 --> U2["Bilinear upsample"]
    D2 -. skip .-> C2["Concat 8w + 4w"]
    U2 --> C2
    C2 --> B2["ConvBlock<br/>12w to 4w"]
    B2 --> U1["Bilinear upsample"]
    D1 -. skip .-> C1["Concat 4w + 2w"]
    U1 --> C1
    C1 --> B1["ConvBlock<br/>6w to 2w"]
    B1 --> U0["Bilinear upsample"]
    F0 -. skip .-> C0["Concat 2w + w"]
    U0 --> C0
    C0 --> B0["ConvBlock<br/>3w to w"]
    B0 --> OUT["Backbone features<br/>width, H x W"]
```

### Prism conditional PatchGAN

Only `prism_gan` creates this discriminator. It judges target or generated
chroma while conditioning on the complete observed input.

```mermaid
flowchart LR
    OBS["Observation [Y, Cr, Cb]<br/>3 channels"] --> CAT["Concatenate"]
    CAN["Candidate [Cr, Cb]<br/>2 channels"] --> CAT
    CAT --> X["5 channels"]
    X --> C1["3 x 3 conv, stride 2<br/>5 to 32 + LeakyReLU"]
    C1 --> C2["3 x 3 conv, stride 2<br/>32 to 64 + LeakyReLU"]
    C2 --> C3["3 x 3 conv, stride 2<br/>64 to 128 + LeakyReLU"]
    C3 --> OUT["3 x 3 conv, stride 1<br/>128 to 1 patch logits"]
```

## Li2026 neural hybrid

The hybrid first constructs an RGB baseline with the audited Li et al.-inspired
analytic pipeline, then lets a frozen-checkpoint-initialized V6 or Prism model
propose a chroma-only correction. A learned scalar gain starts at zero. Retained
sample sites, and optionally locally flat-chromaticity regions, are protected.

```mermaid
flowchart TB
    RGB["Source RGB uint8"] --> ANALYTIC["Li2026 analytic encode/decode<br/>transform + 4:2:0 + interpolation"]
    ANALYTIC --> BASE["Floating RGB baseline"]
    ANALYTIC --> KEEP["Retained-site mask"]
    BASE --> ADAPT["RGB to normalized model YCrCb"]
    ADAPT --> BACKBONE{"Selected backbone"}
    BACKBONE -- "V6" --> V6["V6 residual refiner"]
    BACKBONE -- "Prism" --> PRISM["Prism refiner<br/>with known degradation spec"]
    V6 --> DIFF["Predicted chroma - input chroma"]
    PRISM --> DIFF
    DIFF --> RGBDELTA["Exact zero-luma chroma-to-RGB correction"]
    KEEP --> MASK["Zero protected pixels"]
    BASE -. optional flat-chromaticity guard .-> MASK
    RGBDELTA --> MASK
    MASK --> GAIN["Multiply by tanh of learned scalar gain"]
    BASE --> ADD["Add"]
    GAIN --> ADD
    ADD --> OUT["Refined RGB<br/>retained sites unchanged"]
```

## Hosted TabPFN per-image regressors

The repository owns the feature and reconstruction pipeline, but the selected
hosted `TabPFNRegressor` is an external black box. The diagram deliberately does
not claim internal layers that the repository cannot inspect.

```mermaid
flowchart TB
    X["Target image YCrCb"] --> Y["Full-resolution luma Y"]
    X --> C["Full-resolution target chroma<br/>used only to simulate observations"]
    Y --> FEAT["16 per-pixel features<br/>position, polynomials, sin/cos,<br/>luma, blurs, residual, gradients"]
    FEAT --> FULLX["Full-resolution test rows"]
    FEAT --> LOWX["Area-downsampled training rows"]
    C --> LOWC["Area-downsampled observed chroma"]
    LOWX --> CR["Hosted TabPFNRegressor<br/>fit Cr"]
    LOWC --> CR
    LOWX --> CB["Hosted TabPFNRegressor<br/>fit Cb"]
    LOWC --> CB
    FULLX --> CR
    FULLX --> CB
    CR --> PCR["Predicted full-resolution Cr"]
    CB --> PCB["Predicted full-resolution Cb"]
    Y --> JOIN["Concatenate and clip chroma"]
    PCR --> JOIN
    PCB --> JOIN
    JOIN --> OUT["Reconstructed YCrCb"]
```

## Experiment-to-diagram map

| Experiment group | Forward architecture above |
| --- | --- |
| V5, V5.1, `v5_unet_gan` | V5 U-Net generator; adversarial runs also use V5 PatchGAN |
| `v5_prism_mse` | V5 U-Net generator followed by observed-Y passthrough |
| V6, `v6_residual` | V6 residual chroma refiner |
| `base_v6`, `base_v6_retrained`, `no_luma`, `direct_chroma`, `no_block_skips`, `depth_4`, `depth_12`, `width_32`, `width_96`, `loss_mse`, `loss_charbonnier`, `loss_l1_edge`, `v6_cartesian_deterministic` | Configurable residual ablation family |
| `srcnn_chroma` | Chroma SRCNN |
| `naf_style_chroma` | NAF-style chroma refiner |
| `v7_polar_deterministic`, `v7_probabilistic`, `v7_probabilistic_forward`, `v7_probabilistic_safe`, `v7_self_train` | V7 probabilistic polar refiner |
| `prism_residual`, `prism_mse`, `prism_edge`, `prism_forward` | Prism with residual backbone and Cartesian head |
| `prism_naf` | Prism with NAF backbone and Cartesian head |
| `prism_polar` | Prism with residual backbone and deterministic polar head |
| `prism_polar_prob`, `prism_polar_forward` | Prism with residual backbone, polar head, and uncertainty head |
| `prism_cartesian_prob` | Prism with residual backbone, Cartesian head, and detached uncertainty head |
| `prism_conditioned` | Conditioned Prism with residual backbone and Cartesian head |
| `prism_unet`, `prism_unet_mse` | Prism U-Net with Cartesian head |
| `prism_gan` | Prism U-Net plus conditional Prism PatchGAN during training |
| Li2026 hybrid runs | Li2026 neural hybrid with the named V6 or Prism backbone |
| TabPFN pilot | Hosted TabPFN per-image regressor pipeline |
