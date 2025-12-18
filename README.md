# 🌐 Mapping the "Cognitive Infrastructure" of Global AI Governance
### A Comparative Semantic Network Analysis of US, EU, and CN Policy Discourses

 
## 📖 Introduction & Research Background

As Artificial Intelligence systems scale, the global regulatory landscape is fracturing into distinct geopolitical regimes. While qualitative studies often categorize these regimes broadly: the EU's "Rights-Based" approach, the US's "Market-Driven" approach, and China's "State-Centric" approach. Yet, there still is a lack of quantitative evidence revealing how these ideologies are **structurally embedded** within policy texts.

This project employs **Computational Social Science** methods, specifically **Semantic Network Analysis (SNA)** and **Natural Language Processing (NLP)**, to deconstruct and visualize the "cognitive maps" of policymakers. By treating policy documents not merely as text but as relational data, this research uncovers the latent structural connections between key governance concepts (e.g., *Risk*, *Rights*, *Security*, *Innovation*) across three dominant jurisdictions.

The core research question driving this project is: **How does the semantic structure of AI policy discourse differ across the US, EU, and China, and what do these topological differences reveal about their underlying governance logics?**


## 🔮 Methodology

This project implements a rigorous, data-driven pipeline to transform unstructured policy texts into structured network graphs.

### 1. Data Corpus Construction
The analysis is based on the most recent authoritative documents and frameworks (2024-2025):
* **🇪🇺 European Union:** *The EU AI Act* (Final Compromise Text).
* **🇺🇸 United States:** *Blueprint for an AI Bill of Rights*, *NIST AI Risk Management Framework (RMF)*, *Executive Order 14110*.
* **🇨🇳 China:**
    * **State Council Opinion on Deepening the "AI+" Action**  — Strategic integration.
    * **Measures for the Labeling of AI-Generated Content** (*人工智能生成合成内容标识办法*) — Operational compliance.
    * **AI Safety Governance Framework 2.0** (*人工智能安全治理框架 2.0*) — Full-lifecycle safety guidelines.
 
### 2. Natural Language Processing (NLP) Pipeline
* **Tokenization & Lemmatization:**
    1. English texts processed via `NLTK` with WordNet lemmatization.
    2. Chinese texts segmented via `Jieba` with a custom dictionary for domain-specific policy terms.
* **Conceptual Consolidation (Phrase Mapping):** To ensure semantic validity, fragmented tokens were merged into coherent concepts (e.g., merging `conformity` + `assessmen` $\rightarrow$ `conformity_assessmemt`).
* **Aggressive Noise Reduction:** A bespoke stop-word list was developed to filter out generic administrative and technical noise (e.g., *input*, *output*, *measure*), ensuring the network represents **substantive policy objects**.

### 3. Network Construction & Metrics
* **Co-occurrence Logic:** A sliding window approach (Window Size = 15) was used to capture local semantic context. An edge is created if two terms appear within the same window.
* **Node Selection:** To avoid the "hairball effect," visualization is restricted to the **Top-40** nodes based on **Degree Centrality** calculated from the full corpus metrics.
* **Community Detection:** The **Greedy Modularity Maximization** algorithm is applied to identify and color-code distinct semantic clusters (communities) within the network.

### 4. Visualization Strategy
* **Force-Directed Layout:** A modified Spring Layout is used with **unweighted repulsion** ($k=5.0$). This strategy visually decouples high-frequency clusters, revealing the distinct backbone structure of each regime without the distortion of excessive edge weights.


## 🔬 Key Findings: Divergent Governance Logics

The semantic network analysis reveals three distinct "Cognitive Maps" of AI governance, differing fundamentally in their **objects of regulation** and **spheres of concern**.

### 🇪🇺 European Union: Centralized Market Gatekeeping
* **Hub-and-Spoke Topology:** The EU network shows a highly centralized structure anchored by three key nodes: **`High-Risk`**, **`AI System`**, and **`Product`**.
* **Bureaucratic Density:** There is a high co-occurrence of compliance-related terminology, including **`Conformity Assessment`**, **`Standard`**, **`Notified Body`**, and **`Technical Documentation`**.
* **Observation:** The structure mirrors industrial safety regulations, where "Fundamental Rights" are structurally linked to "Conformity," implying rights are protected through product standardization. The governance logic is Product Safety Regulation. An AI system is treated like a car or a toy; safety is achieved through rigorous documentation and pre-market certification (Notified Body, Technical Documentation).

<div align="center">
  <img 
    src="https://github.com/user-attachments/assets/97e10d59-4f98-48be-8038-9ebba1662ed4" 
    alt="US_figure" 
    width="800"   >
</div>

### 🇺🇸 United States: Sectoral Granularity and Rights
* **Context-Dependent Terminology:** The network is not centered on abstract "AI" but on **`Automated Systems`** and **`Decision Making`**.
* **Sectoral Clustering:** We observe distinct semantic clusters linking governance terms to specific life sectors, specifically **`Housing`**, **`Employment`**, **`Health`**, and **`Finance`**.
* **Normative Focus:** Terms such as **`Bias`**, **`Discrimination`**, and **`Civil Rights`** appear as primary connectors within these sectoral clusters, indicating that governance is triggered by specific downstream harms rather than general technical risks. The governance logic is Contextual Risk Management. There is no single "AI Law"; instead, governance is distributed across sectors, relying on "soft law" tools (NIST Frameworks, audits) to manage risks in specific use cases.
  
<div align="center">
  <img 
    src="https://github.com/user-attachments/assets/cc4ca194-be0a-4062-85de-112c6c02849b" 
    alt="US_figure" 
    width="800"   >
</div>
  

### 🇨🇳 China: Vertical Integration of Infrastructure and Security
* **Infrastructure as Governance Objects:** Unlike Western networks, the Chinese graph exhibits high centrality for upstream technical terms including **`Algorithm (算法)`**, **`Model (模型)`**, **`Data (数据)`**, and **`Computing Power (算力)`**.
* **The "Security" Super-Node:** The term **`Security (安全)`** acts as the network's structural anchor (highest betweenness centrality). It acts as a ubiquitous bridge, connecting developmental nodes (**`Innovation`**) directly with regime-stability nodes (**`Social Order`**, **`State`**).
* **Observation:** The network structure is dense and integrated, suggesting no separation between technical development and state security. The governance logic is Cognitive Security & Information Control.   The primary concern is not just the technical reliability of the system, but the social impact of the information it generates.

<div align="center">
  <img 
    src="https://github.com/user-attachments/assets/484810a0-ac90-4e0a-90fb-319391e7dca7" 
    alt="US_figure" 
    width="800"   >
    </div>

 ### 🌍 Conclusion:
 **The "Safety" Divergence:**
* In the EU: Safety = Product Conformity (Is the technical documentation correct? Is it CE marked?)
* In China: Safety = Ideological/Content Stability (Is the generated content true and socially positive? Does it carry the correct watermarking/labeling?)
* In the US: Safety = System Reliability (Does the system perform accurately in this specific context without bias?)

**Incommensurable Units of Governance:**
* The EU regulates the "System" (categorizing it as High-Risk).
* The US regulates the "Deployment" (impact on Civil Rights in specific sectors).
* China regulates the "Service" (the provider's responsibility for content).

Global harmonization is struggling not merely due to political will, but because the "Cognitive Maps" are misaligned. A "Safety Treaty" signed by all three would fail in practice because the underlying network structures—the way these concepts are operationally implemented—do not map onto each other. Effective coordination requires moving beyond shared vocabulary to structural interoperability, acknowledging that "Safety" in Brussels implies a checklist, while in Beijing it implies a content filter.


## 🚀 Contributions

This project advances the field of Comparative AI Governance through the following theoretical and methodological contributions:
The "Horizontal Safety" Model (EU):** A model that utilizes the logic of Product Safety legislation to create a unified, ex-ante market access regime.

###  Methodological Contribution: Quantitative Verification of Policy Discourse
Existing literature on AI governance often relies on qualitative readings of legal texts. This project contributes **methodological rigor** by:
* **Operationalizing Discourse:** Using Natural Language Processing (NLP) to quantify the "weight" of policy concepts.
* **Visualizing Governance Logic:** Demonstrating that governance philosophy is visible in the *topology* of semantic networks (e.g., the centrality of "Risk" vs. "Security"), providing a reproducible metric for comparative analysis.

### Policy Insight: The "Object" Divergence
We contribute to the debate on global interoperability by identifying that the friction between regimes is not just about values, but about the **object of regulation**:
* The EU regulates the **Product**.
* The US regulates the **Deployment Outcome**.
* China regulates the **Technical Supply Chain**.
This finding suggests that global harmonization efforts must address these structural incompatibilities in regulatory targets.



## 🐈‍⬛ Thanks for reading ;))
