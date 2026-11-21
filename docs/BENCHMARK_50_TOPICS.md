# 50-Topic Stratified Research Proposal Benchmark (`25 Lit2Test + 25 AlphaIdeaBench`)

- **Total Topics**: `50` (`25` from `Lit2Test v0.2 expansion40` + `25` from `AlphaIdeaBench`, exact 1:1 symmetry)
- **Domain Stratification**: `10` core AI/ML & AI4Science domains × strictly `5` topics per domain
- **Calibration Isolation**: `0 / 16` overlap with `Lit2Test fifth40` judge calibration suite; `0` bad/corrupted titles
- **3-Wave Execution Plan**:
  - **`wave1` (10 Canary Topics = 1 per domain, `5 Lit2Test + 5 AlphaIdeaBench`)**: 100% already have frozen `Direct` + `AI-Scientist-v2` source drafts on disk for rapid validation.
  - **`wave2` (+20 Topics = 2 per domain, cumulative 30 topics)**: Held-out expansion wave.
  - **`wave3` (+20 Topics = 2 per domain, cumulative 50 topics)**: Final completion wave.

## Wave 1 Canary Slice (10 Topics — 1 per Domain)

| Domain | Wave | Source | `task_id` | Cached Hosts | Topic Summary |
| :--- | :---: | :---: | :--- | :--- | :--- |
| 大模型推理、检索增强与长上下文 | `wave1` | `alphaideabench` | `survey_01_llm_reasoning_cot` | `direct,ai_scientist` | Towards Reasoning in Large Language Models: A Survey |
| 多模态、视觉语言与语音理解 | `wave1` | `alphaideabench` | `survey_02_vlm_hallucination` | `direct,ai_scientist` | A Survey on Hallucination in Large Vision-Language Models |
| 扩散、流匹配与生成建模 | `wave1` | `lit2test` | `smallari_expansion40_010_problems_inverse_diffusion_projected_solving__v02_expansion40` | `direct,ai_scientist` | problems, inverse, diffusion, projected, solving (Diffusion State-Guided Projected Gradient for Inverse Problems...) |
| 强化学习、具身智能与自主智能体 | `wave1` | `alphaideabench` | `ws_02_agentic_ai` | `direct,ai_scientist,edara` | Towards Agentic AI for Science: Hypothesis Generation, Comprehension, Quantification, and Validation |
| 可信 AI、安全攻防与公平性 | `wave1` | `lit2test` | `smallari_expansion40_002_unlearning_machine_language_use_algorithms__v02_expansion40` | `direct,ai_scientist,edara` | unlearning, machine, language, use, algorithms (A Closer Look at Machine Unlearning for Large Language Models...) |
| 测试时自适应、持续学习与分布外鲁棒性 | `wave1` | `lit2test` | `smallari_expansion40_011_test_time_adaptation_active_tta_test__v02_expansion40` | `direct,ai_scientist` | test-time, adaptation, active, tta, test (Active Test-Time Adaptation: Theoretical Analyses and An Algorith...) |
| 高效推理、模型压缩与机器学习系统 | `wave1` | `alphaideabench` | `survey_08_kv_cache_compression` | `direct,ai_scientist` | Keep the Cost Down: A Review on Methods to Optimize LLM' s KV-Cache Consumption |
| 图神经网络、几何对称性与时序建模 | `wave1` | `lit2test` | `smallari_expansion40_008_graph_networks_message_passing_representation__v02_expansion40` | `direct,ai_scientist` | graph, networks, message, passing, representation (Scalable Message Passing Neural Networks: No Need for Attention i...) |
| 机制可解释性、因果推断与学习理论 | `wave1` | `alphaideabench` | `survey_05_mechanistic_interpretability` | `direct,ai_scientist` | A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models |
| 科学智能（物理/量子/生物分子/临床医疗） | `wave1` | `lit2test` | `smallari_expansion40_003_quantum_circuits_algorithms_design_parameterized__v02_expansion40` | `direct,ai_scientist` | quantum, circuits, algorithms, design, parameterized (Provably Noise-Resilient Training of Parameterized Quantum Circui...) |

## Full 50-Topic Master Table (by Domain)

### D01_llm_reasoning_rag: LLM Reasoning, RAG & Long-Context（大模型推理、检索增强与长上下文）

| # | Wave | Source | Prior Cohort | `task_id` | Topic Summary |
|---:|:---:|:---:|:---:|---|---|
| 1 | `wave1` | `alphaideabench` | `ext15` | `survey_01_llm_reasoning_cot` | Towards Reasoning in Large Language Models: A Survey |
| 2 | `wave2` | `lit2test` | `v4` | `smallari_expansion40_001_cache_compression_llms_long_context__v02_expansion40` | cache, compression, llms, long, context (PyramidKV: Dynamic KV Cache Compression based on Pyramidal Inform...) |
| 3 | `wave2` | `alphaideabench` | `ext15` | `survey_03_rag_systems` | Retrieval-Augmented Generation for Large Language Models: A Survey |
| 4 | `wave3` | `lit2test` | `new25` | `smallari_expansion40_033_retrieval_generation_retrieval_augmented_languag__v02_expansion40` | retrieval, generation, retrieval-augmented, language, augmented (Corrective Retrieval Augmented Generation...) |
| 5 | `wave3` | `alphaideabench` | `new25` | `survey_07_test_time_scaling` | Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters |

### D02_multimodal_vlm_speech: Multimodal, Vision-Language & Speech（多模态、视觉语言与语音理解）

| # | Wave | Source | Prior Cohort | `task_id` | Topic Summary |
|---:|:---:|:---:|:---:|---|---|
| 6 | `wave1` | `alphaideabench` | `ext15` | `survey_02_vlm_hallucination` | A Survey on Hallucination in Large Vision-Language Models |
| 7 | `wave2` | `alphaideabench` | `v4` | `ws_09_world_models_simulation` | Workshop on World Models: Understanding, Modelling and Scaling |
| 8 | `wave2` | `lit2test` | `new25` | `smallari_expansion40_025_audio_visual_multimodal_aligned_language_visual__v02_expansion40` | audio-visual, multimodal, aligned, language, visual (Aligned Better, Listen Better for Audio-Visual Large Language Mod...) |
| 9 | `wave3` | `lit2test` | `new25` | `smallari_expansion40_018_speech_continuation_language_llm_potential__v02_expansion40` | speech, continuation, language, llm, potential (BLSP: Bootstrapping Language-Speech Pre-training via Behavior Ali...) |
| 10 | `wave3` | `alphaideabench` | `new25` | `ws_08_video_language_models` | Workshop on Video-Language Models |

### D03_diffusion_generative: Diffusion, Flow & Generative Modeling（扩散、流匹配与生成建模）

| # | Wave | Source | Prior Cohort | `task_id` | Topic Summary |
|---:|:---:|:---:|:---:|---|---|
| 11 | `wave1` | `lit2test` | `ext15` | `smallari_expansion40_010_problems_inverse_diffusion_projected_solving__v02_expansion40` | problems, inverse, diffusion, projected, solving (Diffusion State-Guided Projected Gradient for Inverse Problems...) |
| 12 | `wave2` | `lit2test` | `new25` | `smallari_expansion40_013_motion_generation_human_editing_flow__v02_expansion40` | motion, generation, human, editing, flow (Motion Flow Matching for Efficient Human Motion Synthesis and Edi...) |
| 13 | `wave2` | `alphaideabench` | `new25` | `survey_14_diffusion_3d_generation` | Diffusion Models in 3D Vision: A Survey |
| 14 | `wave3` | `lit2test` | `new25` | `smallari_expansion40_020_transport_optimal_generative_map_networks__v02_expansion40` | transport, optimal, generative, map, networks (Analyzing and Improving Optimal-Transport-based Adversarial Netwo...) |
| 15 | `wave3` | `alphaideabench` | `new25` | `ws_11_diffusion_models` | Workshop on Diffusion Models |

### D04_rl_robotics_agents: Reinforcement Learning, Robotics & Agents（强化学习、具身智能与自主智能体）

| # | Wave | Source | Prior Cohort | `task_id` | Topic Summary |
|---:|:---:|:---:|:---:|---|---|
| 16 | `wave1` | `alphaideabench` | `v4` | `ws_02_agentic_ai` | Towards Agentic AI for Science: Hypothesis Generation, Comprehension, Quantification, and Validation |
| 17 | `wave2` | `lit2test` | `v4` | `smallari_expansion40_040_critic_reinforcement_actor_critic_actor_continuo__v02_expansion40` | critic, reinforcement, actor-critic, actor, continuous (Afterstate Reinforcement Learning for Continuous Control...) |
| 18 | `wave2` | `alphaideabench` | `new25` | `survey_04_llm_agent_collaboration` | A Survey on Large Language Model based Autonomous Agents |
| 19 | `wave3` | `alphaideabench` | `new25` | `survey_10_embodied_ai_vla` | A Survey on Vision-Language-Action Models for Embodied AI |
| 20 | `wave3` | `alphaideabench` | `new25` | `ws_35_real_world_rl` | Reincarnating RL |

### D05_trustworthy_safety: Trustworthy AI, Safety & Fairness（可信 AI、安全攻防与公平性）

| # | Wave | Source | Prior Cohort | `task_id` | Topic Summary |
|---:|:---:|:---:|:---:|---|---|
| 21 | `wave1` | `lit2test` | `v4` | `smallari_expansion40_002_unlearning_machine_language_use_algorithms__v02_expansion40` | unlearning, machine, language, use, algorithms (A Closer Look at Machine Unlearning for Large Language Models...) |
| 22 | `wave2` | `alphaideabench` | `v4` | `survey_06_llm_safety_jailbreak` | Jailbreak Attacks and Defenses Against Large Language Models: A Survey |
| 23 | `wave2` | `lit2test` | `ext15` | `smallari_expansion40_006_backdoor_attacks_perspective_attack_triggers__v02_expansion40` | backdoor, attacks, perspective, attack, triggers (Demystifying Poisoning Backdoor Attacks from a Statistical Perspe...) |
| 24 | `wave3` | `lit2test` | `ext15` | `smallari_expansion40_007_conformal_prediction_sets_provably_efficiency__v02_expansion40` | conformal, prediction, sets, provably, efficiency (CBMA: Improving Conformal Prediction through Bayesian Model Avera...) |
| 25 | `wave3` | `lit2test` | `ext15` | `smallari_expansion40_017_group_fairness_individual_in_processing_fair__v02_expansion40` | group, fairness, individual, in-processing, fair (FFB: A Fair Fairness Benchmark for In-Processing Group Fairness M...) |

### D06_adaptation_continual_ood: Adaptation, Continual & OOD Robustness（测试时自适应、持续学习与分布外鲁棒性）

| # | Wave | Source | Prior Cohort | `task_id` | Topic Summary |
|---:|:---:|:---:|:---:|---|---|
| 26 | `wave1` | `lit2test` | `v4` | `smallari_expansion40_011_test_time_adaptation_active_tta_test__v02_expansion40` | test-time, adaptation, active, tta, test (Active Test-Time Adaptation: Theoretical Analyses and An Algorith...) |
| 27 | `wave2` | `lit2test` | `ext15` | `smallari_expansion40_009_detection_out_of_distribution_ood_post_hoc_in_di__v02_expansion40` | detection, out-of-distribution, ood, post-hoc, in-distribution (Semantic or Covariate? A Study on the Intractable Case of Out-of-...) |
| 28 | `wave2` | `lit2test` | `ext15` | `smallari_expansion40_014_spurious_correlations_features_networks_activati__v02_expansion40` | spurious, correlations, features, networks, activations (The Invariance Starvation Hypothesis...) |
| 29 | `wave3` | `alphaideabench` | `new25` | `ws_44_distribution_shifts` | Workshop on Distribution Shifts: New Frontiers with Foundation Models |
| 30 | `wave3` | `alphaideabench` | `new25` | `survey_09_llm_continual_learning` | Continual Learning of Large Language Models: A Comprehensive Survey |

### D07_efficiency_systems: Efficiency, Compression & ML Systems（高效推理、模型压缩与机器学习系统）

| # | Wave | Source | Prior Cohort | `task_id` | Topic Summary |
|---:|:---:|:---:|:---:|---|---|
| 31 | `wave1` | `alphaideabench` | `v4` | `survey_08_kv_cache_compression` | Keep the Cost Down: A Review on Methods to Optimize LLM' s KV-Cache Consumption |
| 32 | `wave2` | `lit2test` | `ext15` | `smallari_expansion40_023_quantization_quantization_aware_fine_tuning_diff__v02_expansion40` | quantization, quantization-aware, fine-tuning, diffusion, post-training (AQUATIC-Diff: Additive Quantization for Truly Tiny Compressed Dif...) |
| 33 | `wave2` | `lit2test` | `new25` | `smallari_expansion40_030_pruning_language_llms_however_layer_wise__v02_expansion40` | pruning, language, llms, however, layer-wise (RotPruner: Large Language Model Pruning in Rotated Space...) |
| 34 | `wave3` | `lit2test` | `new25` | `smallari_expansion40_026_low_rank_full_rank_llms_fine_tuning_memory__v02_expansion40` | low-rank, full-rank, llms, fine-tuning, memory (AdaRankGrad: Adaptive Gradient Rank and Moments for Memory-Effici...) |
| 35 | `wave3` | `alphaideabench` | `new25` | `survey_18_speculative_decoding` | Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding |

### D08_graph_geometric_temporal: Graph, Geometric & Temporal ML（图神经网络、几何对称性与时序建模）

| # | Wave | Source | Prior Cohort | `task_id` | Topic Summary |
|---:|:---:|:---:|:---:|---|---|
| 36 | `wave1` | `lit2test` | `ext15` | `smallari_expansion40_008_graph_networks_message_passing_representation__v02_expansion40` | graph, networks, message, passing, representation (Scalable Message Passing Neural Networks: No Need for Attention i...) |
| 37 | `wave2` | `lit2test` | `ext15` | `smallari_expansion40_012_series_time_anomaly_anomalies_detection__v02_expansion40` | series, time, anomaly, anomalies, detection (When Will It Fail?: Anomaly to Prompt for Forecasting Future Anom...) |
| 38 | `wave2` | `alphaideabench` | `new25` | `ws_38_temporal_graph_learning` | Temporal Graph Learning Workshop |
| 39 | `wave3` | `lit2test` | `new25` | `smallari_expansion40_034_equivariant_networks_breaking_symmetry_simple__v02_expansion40` | equivariant, networks, breaking, symmetry, simple (Improving Equivariant Networks with Probabilistic Symmetry Breaki...) |
| 40 | `wave3` | `alphaideabench` | `new25` | `survey_24_graph_foundation_models` | Graph Foundation Models: Concepts, Opportunities and Challenges |

### D09_interpretability_causality_theory: Interpretability, Causality & Theory（机制可解释性、因果推断与学习理论）

| # | Wave | Source | Prior Cohort | `task_id` | Topic Summary |
|---:|:---:|:---:|:---:|---|---|
| 41 | `wave1` | `alphaideabench` | `ext15` | `survey_05_mechanistic_interpretability` | A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models |
| 42 | `wave2` | `lit2test` | `v4` | `smallari_expansion40_029_bottleneck_concept_concepts_information_cbms__v02_expansion40` | bottleneck, concept, concepts, information, cbms (Zero-shot Concept Bottleneck Models via Sparse Regression of Retr...) |
| 43 | `wave2` | `lit2test` | `new25` | `smallari_expansion40_005_memorization_deep_generalization_necessary_funda__v02_expansion40` | memorization, deep, generalization, necessary, fundamentals (Back to Fundamentals: Re-Examining Memorization in Deep Learning ...) |
| 44 | `wave3` | `alphaideabench` | `new25` | `ws_50_causal_learning_theory` | Causality and Large Models |
| 45 | `wave3` | `alphaideabench` | `new25` | `survey_46_causal_representation_learning` | Towards Causal Representation Learning |

### D10_ai_for_science_biomed: AI for Science & Biomedicine（科学智能（物理/量子/生物分子/临床医疗））

| # | Wave | Source | Prior Cohort | `task_id` | Topic Summary |
|---:|:---:|:---:|:---:|---|---|
| 46 | `wave1` | `lit2test` | `ext15` | `smallari_expansion40_003_quantum_circuits_algorithms_design_parameterized__v02_expansion40` | quantum, circuits, algorithms, design, parameterized (Provably Noise-Resilient Training of Parameterized Quantum Circui...) |
| 47 | `wave2` | `lit2test` | `ext15` | `smallari_expansion40_019_protein_language_structural_structure_sequence__v02_expansion40` | protein, language, structural, structure, sequence (Endowing Protein Language Models with Structural Knowledge...) |
| 48 | `wave2` | `alphaideabench` | `v4` | `ws_18_scientific_foundation_models` | Workshop on Open Science for Foundation Models |
| 49 | `wave3` | `alphaideabench` | `new25` | `ws_14_physics_for_ml` | Physics for Machine Learning |
| 50 | `wave3` | `alphaideabench` | `new25` | `survey_30_medical_vlm_healthcare` | Medical Multimodal Foundation Models in Clinical Diagnosis and Treatment: Applications, Challenges, and Future Directions |

