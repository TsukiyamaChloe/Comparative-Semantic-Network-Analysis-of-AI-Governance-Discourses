# A Comparative Semantic Network Analysis of US, EU, and CN Policy Discourses

 
## 📖 Introduction & Research Background

As Artificial Intelligence systems scale, the global regulatory landscape is fracturing into distinct geopolitical regimes. While qualitative studies often categorize these regimes broadly: the EU's "Rights-Based" approach, the US's "Market-Driven" approach, and China's "State-Centric" approach. Yet, there still is a lack of quantitative evidence revealing how these ideologies are **structurally embedded** within policy texts.

This project employs **Computational Social Science** methods, specifically **Semantic Network Analysis (SNA)** and **Natural Language Processing (NLP)**, to deconstruct and visualize the "cognitive maps" of policymakers. By treating policy documents not merely as text but as relational data, this research uncovers the latent structural connections between key governance concepts (e.g., *Risk*, *Rights*, *Security*, *Innovation*) across three dominant jurisdictions.

The core research question driving this project is: **How does the semantic structure of AI policy discourse differ across the US, EU, and China, and what do these topological differences reveal about their underlying governance logics?**

## 🔸 Methodology

In this project, I developed a custom **Computational Social Science** pipeline to transform unstructured policy documents into structured semantic networks. The analytical framework proceeds through three distinct phases: ETL (Extract, Transform, Load), Network Topology Modeling, and Visualization.

### 1. Corpus Construction & Preprocessing
I processed the raw textual data using a custom ETL script (`corpus_builder.py`) designed to handle the idiosyncrasies of policy PDFs:

* **Data Corpus Construction**
The analysis is based on the most recent authoritative documents and frameworks (2024-2025):
  * **🇪🇺 European Union:** *The EU AI Act* (Final Compromise Text).
  * **🇺🇸 United States:** *Blueprint for an AI Bill of Rights*, *NIST AI Risk Management Framework (RMF)*, *Executive Order 14110*.
  * **🇨🇳 China:**
      **State Council Opinion on Deepening the "AI+" Action**  — Strategic integration.
      **Measures for the Labeling of AI-Generated Content** (*人工智能生成合成内容标识办法*) — Operational compliance.
      **AI Safety Governance Framework 2.0** (*人工智能安全治理框架 2.0*) — Full-lifecycle safety guidelines.

* **Text Extraction & OCR Repair:** I parsed raw PDF documents using `pdfplumber`, applying region-specific cropping to exclude headers and footers. To address OCR artifacts common in government documents, I implemented Regex-based cleaning rules to reconstruct broken tokens (e.g., repairing `"ar tificial"` $\rightarrow$ `"artificial"`).

* **NLP Tokenization:**
    * **English (US/EU):** Processed via `NLTK` with WordNet Lemmatization to standardize word forms.
    * **Chinese (CN):** Processed via `Jieba` for precise word segmentation.
* **Phrase Mapping:** To preserve semantic integrity, I consolidated multi-word expressions into single tokens (e.g., *"automated decision making"* $\rightarrow$ `automated_decision`) before analysis.
  
* **Noise Reduction:** I applied a rigorous, domain-specific stop-word filter to remove generic administrative terminology (e.g., *measure*, *article*, *input*, *output*), ensuring that the resulting nodes represent substantive policy concepts.

### 2. Network Topology Modeling
I constructed the semantic networks based on statistical co-occurrence metrics (`metrics_calculator.py`):

* **Node Selection:** To mitigate the "hairball effect" common in large-scale text networks, I isolated the **Top-K (N=55)** concepts based on **Degree Centrality**. This ensures the graph highlights the most structurally significant terms in the discourse rather than merely the most frequent ones.
  
* **Edge Definition (Sliding Window):** I defined edges using a **Sliding Window Algorithm** (Window Size = 10-15 tokens). If two valid concepts co-occur within this proximity, an edge is established. This method captures local semantic context rather than global document frequency.
  
* **Metric Calculation:** I computed **Degree Centrality** to measure node prominence and **Betweenness Centrality** to identify structural "bridges" that connect disparate semantic clusters.

### 3. Visualization & Community Detection
The final visualization (`figures_pipeline.py`) utilizes Graph Theory algorithms to reveal latent governance logic:

* **Community Detection:** I applied the **Greedy Modularity Maximization** algorithm to mathematically partition the network into dense communities. These clusters are color-coded to reveal latent thematic subgroups (e.g., a "Rights" cluster vs. a "Safety" cluster).
  
* **Force-Directed Layout:** I rendered the networks using the **Fruchterman-Reingold (Spring)** layout. Crucially, I applied **unweighted repulsion** with a high spacing factor ($k$) to push unconnected nodes apart while keeping semantic neighbors close. This effectively "explodes" dense cores to reveal their internal topology without visual overlapping.

## 🔹 Key Findings: Divergent Governance Logics

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

 ### 🔺 Conclusion:
 **The "Safety" Divergence:**
* In the EU: Safety = Product Conformity (Is the technical documentation correct? Is it CE marked?)
* In China: Safety = Ideological/Content Stability (Is the generated content true and socially positive? Does it carry the correct watermarking/labeling?)
* In the US: Safety = System Reliability (Does the system perform accurately in this specific context without bias?)

**Incommensurable Units of Governance:**
* The EU regulates the **industrial Product** (categorizing it as High-Risk).
* The US regulates the **Deployment consequence** (impact on Civil Rights in specific sectors).
* China regulates the **Technical Supply Chaine** (the provider's responsibility for content).

In Conclusion, global harmonization is struggling not merely due to political will, but because the "Cognitive Maps" are misaligned. A "Safety Treaty" signed by all three would fail in practice because the underlying network structures, the way these concepts are operationally implementedo, not map onto each other. Effective coordination requires moving beyond shared vocabulary to structural interoperability, acknowledging that "Safety" in Brussels implies a checklist, while in Beijing it implies a content filter.


### ▪️ Reproducibility

Follow these steps to replicate the data processing and visualization pipeline.

#### 1. Prerequisites
Ensure you have **Python 3.9+** installed. Install the required dependencies in the requirement.txt.

#### 2. Execution Pipeline
Run the scripts in the following order to reproduce the results:
* **Step 1**: Corpus Extraction Extracts text from PDFs, cleans artifacts, and merges them into region-specific datasets. You can replace the pdfs in raw_pdfs with other regulations documents to compare different laws. (Output: Cleaned .txt files in processed_txt/)
  
* **Step 2**: Metrics Calculation Performs tokenization, calculates degree centrality, and generates co-occurrence matrices. (Output: _metrics.csv files in quantitative_data/.)

* **Step 3**: Network Visualization Generates the semantic network graphs. This script automatically handles font detection for Chinese characters (SimHei/Arial Unicode) and applies community detection algorithms. (output:.jpg network graphs in figures/.)
            
* **Step 4**: Summary Generation Produces the comparative data table used in the final report. (output: Table.csv containing the comparative rankings of top policy terms.)

          
## 🐈‍⬛ Thanks for reading ;))
