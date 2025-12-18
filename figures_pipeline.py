import os
import re
import platform
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import jieba
from nltk.stem import WordNetLemmatizer
from networkx.algorithms import community

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_DIR = os.path.join(BASE_DIR, 'processed_txt')
METRICS_DIR = os.path.join(BASE_DIR, 'quantitative_data') 
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

if platform.system() == 'Darwin':  
    CHINESE_FONT = 'Arial Unicode MS' 
    ENGLISH_FONT = 'Arial'
else:  
    CHINESE_FONT = 'SimHei'
    ENGLISH_FONT = 'Arial'

TOP_K_NODES = 55 
WINDOW_SIZE = 10


MANDATORY_CONCEPTS = {
    'CN': {'国家', '攻击', '创新', '社会', '安全'}, 
    'US': set(),
    'EU': set()
}

STOP_WORDS_EN = {
    'system', 'systems', 'automated', 'process', 'ensure', 'including', 
    'use', 'used', 'using', 'provider', 'shall', 'may', 'article', 
    'paragraph', 'union', 'member', 'state', 'requirement', 'measure',
    'input', 'output', 'content', 'application', 'level', 'list', 'based',
    'approach', 'part', 'set', 'related', 'datum', 'data' 
}

STOP_WORDS_CN = {
    '应当', '不仅', '以及', '为了', '方面', '包括', '与其', '相关', '具有', '可以',
    '或者', '工作', '提供', '情况', '对于', '要求', '进行', '重大', '重要'
}

PHRASE_MAPPINGS_EN = [
    (r'\bautomated[\s-]systems?\b', 'automated_system'),
    (r'\bautomated[\s-]decision[\s-]making\b', 'automated_decision'),
    (r'\bai[\s-]systems?\b', 'ai_system'),
    (r'\bgenerative[\s-]ai\b', 'generative_ai'),
    (r'\bgeneral[\s-]purpose[\s-]ai\b', 'general_purpose_ai'),
    (r'\bhigh[\s-]risk\b', 'high_risk'),
    (r'\bfundamental[\s-]rights?\b', 'fundamental_rights'),
    (r'\bartificial[\s-]intelligence\b', 'ai'),
    (r'\bnational[\s-]security\b', 'national_security'),
    (r'\bsupply[\s-]chains?\b', 'supply_chain'),
]

class SemanticNetworkAnalysis:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def load_nodes_from_metrics(self, csv_path, region, top_k=TOP_K_NODES):

        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            
            if 'Term' in df.columns:
                if region == 'CN':
                    df = df[~df['Term'].isin(STOP_WORDS_CN)]
                else:
                    df = df[~df['Term'].isin(STOP_WORDS_EN)]
            
            df_sorted = df.sort_values(by='Degree_Centrality', ascending=False)
            top_nodes = set(df_sorted.head(top_k)['Term'].astype(str).tolist())
            
            mandatory = MANDATORY_CONCEPTS.get(region, set())
            valid_mandatory = set()
            
            if mandatory:
                all_terms = set(df['Term'].astype(str).tolist())
                for term in mandatory:
                    if term in all_terms:
                        valid_mandatory.add(term)
                    else:
                        print(f"Warning: Mandatory term '{term}' not found in {csv_path}")
            
            final_nodes = top_nodes.union(valid_mandatory)
            
            df_final = df[df['Term'].isin(final_nodes)]
            sizes = df_final['Degree_Centrality'].values
            
            if len(sizes) > 0 and sizes.max() != sizes.min():
                norm = (sizes - sizes.min()) / (sizes.max() - sizes.min())
                size_map = dict(zip(df_final['Term'], norm * 3000 + 800))
            else:
                size_map = dict(zip(df_final['Term'], [1500]*len(sizes)))
                
            print(f"[{region}] Selected {len(final_nodes)} nodes (Top {top_k} + {len(valid_mandatory)} Mandatory).")
            return final_nodes, size_map
            
        except Exception as e:
            print(f"Error loading CSV {csv_path}: {e}")
            return set(), {}

    def preprocess_text(self, text, lang='en'):
        text = text.lower()
        if lang == 'en':
            for pattern, repl in PHRASE_MAPPINGS_EN:
                text = re.sub(pattern, repl, text)
            tokens = re.findall(r'\b[a-zA-Z_]+\b', text)
            clean_tokens = []
            for t in tokens:
                lemma = t if '_' in t else self.lemmatizer.lemmatize(t)
                if lemma not in STOP_WORDS_EN and len(lemma) > 2:
                    clean_tokens.append(lemma)
            return clean_tokens
        elif lang == 'cn':
            tokens = jieba.lcut(text)
            return [t for t in tokens if len(t) > 1 and t not in STOP_WORDS_CN]
        return []

    def build_graph(self, tokens, valid_nodes):
        G = nx.Graph()
        G.add_nodes_from(valid_nodes)
        
        for i in range(len(tokens) - WINDOW_SIZE + 1):
            window = tokens[i : i + WINDOW_SIZE]
            relevant = [w for w in window if w in valid_nodes]
            for j in range(len(relevant)):
                for k in range(j + 1, len(relevant)):
                    u, v = relevant[j], relevant[k]
                    if u == v: continue
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += 1
                    else:
                        G.add_edge(u, v, weight=1)
        
        G.remove_nodes_from(list(nx.isolates(G)))
        return G

    def plot_graph(self, G, size_map, title, filename, font):
        plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False 

        plt.figure(figsize=(20, 18))
        
        try:
            communities = community.greedy_modularity_communities(G)
            community_map = {node: i for i, comm in enumerate(communities) for node in comm}
            colors = [community_map.get(n, 0) for n in G.nodes()]
            cmap = plt.cm.get_cmap('tab10', len(communities))
        except:
            colors = '#66c2a5'
            cmap = None


        pos = nx.spring_layout(G, k=5.0, seed=42, iterations=1000, weight=None)
        
        node_sizes = [size_map.get(n, 500) for n in G.nodes()]
        
        edges = G.edges(data=True)
        weights = [d['weight'] for u, v, d in edges] if edges else []
        if weights:
            max_w = max(weights)
            widths = [1.5 + 4.5 * (w / max_w) for w in weights]
            edge_colors = [(0.4, 0.4, 0.4, 0.3 + 0.4 * (w / max_w)) for w in weights]
        else:
            widths, edge_colors = [], []

        nx.draw_networkx_edges(G, pos, width=widths, edge_color=edge_colors)
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                               node_color=colors, cmap=cmap,
                               alpha=0.9, edgecolors='white', linewidths=2.5)
        
        labels = nx.draw_networkx_labels(G, pos, font_family=font, font_size=16, font_weight='bold')
        for txt in labels.values():
            txt.set_path_effects([path_effects.Stroke(linewidth=6, foreground='white', alpha=0.9), path_effects.Normal()])
            
        plt.title(title, fontsize=30, fontname=font, y=0.90)
        
        plt.margins(0.06, 0.06)
        plt.axis('off')
        plt.tight_layout()
        
        save_path = os.path.join(FIGURES_DIR, f"{filename}.jpg")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated {save_path}")

def main():
    sna = SemanticNetworkAnalysis()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    tasks = [
        ('US', 'US_ALL.txt', 'US_metrics.csv', 'en', ENGLISH_FONT),
        ('EU', 'EU_ALL.txt', 'EU_metrics.csv', 'en', ENGLISH_FONT),
        ('CN', 'CN_ALL.txt', 'CN_metrics.csv', 'cn', CHINESE_FONT),
    ]
    
    print(f"Script Location: {BASE_DIR}")
    
    for region, txt_name, csv_name, lang, font in tasks:
        txt_path = os.path.join(TXT_DIR, txt_name)
        csv_path = os.path.join(METRICS_DIR, csv_name)
        
        if not os.path.exists(csv_path):
             csv_path_fallback = os.path.join(BASE_DIR, csv_name)
             if os.path.exists(csv_path_fallback):
                 csv_path = csv_path_fallback

        if not os.path.exists(csv_path):
            print(f"Skipping {region}: CSV not found at {csv_path}")
            continue
            
        print(f"Processing {region}...")
        
        nodes, size_map = sna.load_nodes_from_metrics(csv_path, region, top_k=TOP_K_NODES)
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                tokens = sna.preprocess_text(f.read(), lang)
            G = sna.build_graph(tokens, nodes)
            sna.plot_graph(G, size_map, f"{region} Semantic Network", f"{region}_figure", font)
        else:
            print(f"Warning: Text file {txt_path} not found.")

if __name__ == "__main__":
    main()