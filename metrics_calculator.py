import os
import re
import math
import jieba
import pandas as pd
import networkx as nx
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter, defaultdict

try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

BASE_DIR = os.getcwd()
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_txt')
OUTPUT_DIR = os.path.join(BASE_DIR, 'quantitative_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

lemmatizer = WordNetLemmatizer()

# -----------------------------
# Options
# -----------------------------
WINDOW_SIZE = 15
TOP_N = 50

# If you want the bare token "system" to be kept (NOT recommended for your original goal), set True.
KEEP_BARE_SYSTEM = False

# -----------------------------
# Domain-specific phrases
# -----------------------------
DOMAIN_PHRASES_EN = [
    # Core AI objects
    'artificial intelligence system', 'artificial intelligence systems',
    'ai system', 'ai systems',
    'general purpose ai', 'general-purpose ai',
    'foundation model', 'foundation models',
    'generative model', 'generative models',
    'large language model', 'large language models',
    'machine learning', 'deep learning',
    'automated system', 'automated systems',
    'decision making', 'decision-making',
    'automated decision making', 'automated decision-making',
    'trustworthy ai', 'trustworthy artificial intelligence',

    # Risk and safety
    'risk management', 'risk assessment', 'risk mitigation',
    'safety measure', 'safety measures',
    'systemic risk', 'high risk', 'high-risk',

    # Compliance and regulation interface terms
    'conformity assessment', 'market surveillance',
    'technical documentation', 'technical requirement',
    'fundamental right', 'fundamental rights',
    'transparency requirement', 'transparency requirements',
    'quality management', 'due diligence',

    # EU institutional terms
    'internal market', 'notified body', 'competent authority',
    'natural person', 'biometric identification',
    'placing on the market', 'ai value chain',

    # Miscellaneous
    'information security', 'ai ethics', 'value alignment',
    'personal data', 'data safety',
    'service provider', 'service providers',
    'algorithm recommendation',
    'automated decision', 'automated decisions',
]

DOMAIN_PHRASES_CN = [
    '人工智能', '生成式', '算法推荐', '内容安全', '风险评估',
    '服务提供者', '技术标准', '数据安全', '模型训练',
    '安全评估', '智能化', '生成合成', '标识管理',
    '可信', '可信赖', '可信任', '值得信赖', '可靠'
]

for p in DOMAIN_PHRASES_CN:
    jieba.add_word(p)

# -----------------------------
# Whitelist / stopwords
# -----------------------------
SUBSTANTIVE_WHITELIST = {
    # Core objects
    'ai',
    'risk', 'safety', 'security', 'privacy', 'transparency',
    'accountability', 'bias', 'fairness', 'discrimination',
    'harm', 'vulnerability', 'threat', 'hazard',
    'robustness', 'accuracy', 'performance',

    # Rights and values
    'right', 'freedom', 'dignity', 'autonomy', 'justice',
    'equality', 'diversity', 'inclusion', 'ethics', 'value', 'trustworthy',

    # Human subjects
    'person', 'individual', 'user', 'consumer', 'worker',
    'child', 'minor', 'human', 'people', 'citizen', 'community',

    # Institutional actors
    'authority', 'regulator', 'supervisor', 'auditor',

    # Governance mechanisms
    'responsibility', 'liability', 'duty',
    'compliance', 'penalty', 'sanction',
    'oversight', 'supervision', 'surveillance', 'monitoring',
    'certification', 'registration', 'notification',

    # Technical/product terms
    'algorithm', 'input', 'output', 'training', 'deployment', 'testing',
    'reliability',

    # Market/service
    'service', 'product', 'supply', 'chain',
    'infrastructure', 'platform', 'application',

    # Legal/policy
    'legislation', 'framework',
    'standard', 'guideline', 'principle', 'criterion',

    # Impact domains
    'health', 'education', 'employment', 'finance',
    'welfare', 'environment', 'democracy',

    # Documentation
    'documentation', 'record', 'log', 'report', 'disclosure',
}

AGGRESSIVE_STOPWORDS = {
    # Function words
    'the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'for', 'it', 'as', 'with', 'on', 'be', 'are', 'by',
    'an', 'this', 'these', 'those', 'or', 'which', 'will', 'not', 'can', 'should', 'shall', 'may', 'at',
    'from', 'has', 'have', 'who', 'whom', 'whose', 'where', 'when', 'how', 'why', 'what', 'there', 'here',

    # Pronouns
    'you', 'your', 'we', 'our', 'us', 'they', 'their', 'them', 'its', 'he', 'she',

    # Auxiliary verbs
    'did', 'do', 'does', 'done', 'doing', 'been', 'being', 'had', 'having', 'would', 'could',

    # Determiners and quantifiers
    'such', 'other', 'any', 'all', 'some', 'no', 'nor', 'more', 'most', 'less', 'very',

    # Prepositions
    'about', 'into', 'over', 'under', 'between', 'through', 'during', 'before', 'after', 'above', 'below',

    # Boilerplate verbs/common fillers
    'also', 'within', 'ensure', 'ensuring', 'including', 'include', 'includes', 'included',
    'regarding', 'related', 'relating', 'provide', 'provided', 'providing',
    'use', 'used', 'using', 'uses', 'make', 'made', 'making',
    'create', 'creating', 'take', 'taken', 'taking', 'identify', 'identified',
    'implement', 'implementing', 'affect', 'affecting', 'affects',
    'assess', 'assessing', 'assessed', 'monitor', 'monitoring',
    'comply', 'complying', 'design', 'designing', 'designed',
    'develop', 'developing', 'developed', 'require', 'requiring', 'required',
    'support', 'supporting', 'supported', 'train', 'training', 'trained',
    'carry', 'carrying', 'carried', 'conduct', 'conducting',
    'perform', 'performing', 'enable', 'enabling',
    'consider', 'considering', 'address', 'addressing',

    # Generic adjectives/states
    'appropriate', 'necessary', 'relevant', 'specific', 'specified',
    'following', 'available', 'applicable', 'possible',
    'potential', 'likely', 'certain', 'particular', 'general',
    'legal', 'natural', 'personal', 'public', 'internal',

    # Document structure
    'article', 'paragraph', 'section', 'chapter', 'annex', 'appendix',
    'recital', 'provision', 'provisions', 'text', 'document', 'page', 'date',
    'number', 'version', 'reference', 'definitions',

    # Legal boilerplate
    'accordance', 'pursuant', 'laid', 'down', 'set', 'out',
    'apply', 'applies', 'application', 'establish', 'established',

    # Organizational boilerplate
    'member', 'states', 'state', 'union', 'european', 'commission',
    'council', 'parliament', 'agency', 'bodies', 'body', 'office',
    'board', 'committee', 'united', 'national', 'federal',
    'government', 'entity', 'entities', 'countries', 'country',
    'sector', 'sectors', 'level', 'group', 'groups',

    # Misc policy boilerplate
    'map', 'roadmap', 'blueprint', 'signatory', 'list',
    'without', 'refer', 'reference',
    'place', 'placed', 'placing', 'notify', 'notified', 'notification',
    'follow', 'following', 'regard', 'laying',
    'put', 'mean', 'means', 'relate', 'whether', 'intend', 'request',
    'rule', 'rules', 'case', 'cases', 'subject',
    'context', 'order', 'result', 'approach', 'approaches',
    'base', 'basis', 'only', 'requirement', 'requirements',
}

# Filter these as singleton tokens only. Phrase forms (underscore) are preserved.
CONDITIONAL_SINGLETON_STOPWORDS_EN = {
    "data",
    "model",
    "information",
    "regulation", "law", "authority", "market",
}
if not KEEP_BARE_SYSTEM:
    CONDITIONAL_SINGLETON_STOPWORDS_EN.update({"system", "systems"})

STOPWORDS_CN = {
    '的', '和', '与', '在', '是', '等', '对于', '具有', '或者', '应当', '必须', '进行', '了', '为', '以',
    '及', '其', '中', '对', '将', '不', '可', '由', '向', '上', '并', '各', '该', '有关', '相关',
    '主要', '重要', '加强', '推进', '实施', '完善', '建立', '健全', '工作', '方面', '领域',
    '不仅', '而且', '虽然', '但是', '通过', '根据', '关于', '按照', '以及', '同时', '包括',
    '条', '第', '章', '节', '附件', '办法', '规定', '意见', '通知', '方案', '规划', '行动',
    '发布', '印发', '年月日', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
    '我们', '他们', '你们', '它', '它们', '2.0', '1.0', '3.0',
    '提供', '使用', '能力', '水平',
    '以及', '需要', '可以', '没有', '能够', '为了', '提升', '推动', '完善', '实施',
    '推进', '促进', '具有', '相关', '有关', '工作', '方面', '2025', '2030'
}

# -----------------------------
# Region protected nodes (prevents "anchor disappearance" under NPMI penalties)
# -----------------------------
PROTECTED_BY_REGION = {
    'EU': {
        'ai', 'ai_system', 'general_purpose_ai',
        'risk', 'high_risk', 'systemic_risk',
        'right', 'fundamental_rights',
        'safety', 'security',
        'trustworthy_ai',
    },
    'US': {
        'ai', 'ai_system', 'automated_system',
        'risk', 'right', 'harm', 'privacy',
        'discrimination', 'bias',
        'safety', 'security',
    },
    'CN': {'智能', '安全', '人工智能', '风险', '权利'},
}

# -----------------------------
# Phrase identification & token utils
# -----------------------------
def identify_phrases(text, phrase_list):
    sorted_phrases = sorted(phrase_list, key=lambda x: len(x.split()), reverse=True)
    for phrase in sorted_phrases:
        p = phrase.lower().strip()
        words = p.split()
        sep = r"(?:\s+|-)+"
        if len(words) == 1:
            pattern = rf"\b{re.escape(words[0])}\b"
        else:
            pattern = rf"\b{sep.join(re.escape(w) for w in words)}\b"
        replacement = p.replace(' ', '_').replace('-', '_')
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def normalize_compound_term(token):
    if '_' not in token:
        return token
    parts = token.split('_')
    normalized_parts = [lemmatizer.lemmatize(p, pos='n') for p in parts]
    return '_'.join(normalized_parts)

def clean_token(word):
    word = word.lower().strip()
    lemma = lemmatizer.lemmatize(word, pos='n')
    lemma = lemmatizer.lemmatize(lemma, pos='v')
    return lemma

def apply_canonical_mapping(tokens):
    CANONICAL = {
        "ai_systems": "ai_system",
        "automated_systems": "automated_system",
        "fundamental_right": "fundamental_rights",
        "fundamental_rights": "fundamental_rights",
        "notified_bodies": "notified_body",
        "service_providers": "service_provider",
        "智能": "人工智能",
        # common variants
        "decision_makings": "decision_making",
    }
    return [CANONICAL.get(t, t) for t in tokens]

# -----------------------------
# Token extraction
# -----------------------------
def get_tokens(region):
    path = os.path.join(PROCESSED_DIR, f"{region}_ALL.txt")
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return []

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = []

    if region == 'CN':
        text_phrased = identify_phrases(text.lower(), DOMAIN_PHRASES_CN)
        words = jieba.lcut(text_phrased)
        tokens = [
            w for w in words
            if len(w) > 1
            and w not in STOPWORDS_CN
            and not re.match(r'^[0-9.]+$', w)
        ]
        tokens = apply_canonical_mapping(tokens)
    else:
        text_phrased = identify_phrases(text.lower(), DOMAIN_PHRASES_EN)
        text_phrased = re.sub(r'[^\w\s_-]', '', text_phrased)

        for w in text_phrased.split():
            if '_' in w:
                if len(w) > 2:
                    tokens.append(normalize_compound_term(w))
            else:
                lemma = clean_token(w)

                # roman numerals
                if re.fullmatch(r"[ivxlcdm]+", lemma):
                    continue

                # allow short 'ai' in metrics
                if lemma == 'ai':
                    tokens.append('ai')
                    continue

                if lemma in CONDITIONAL_SINGLETON_STOPWORDS_EN:
                    continue

                if (len(lemma) > 2
                    and lemma not in AGGRESSIVE_STOPWORDS
                    and lemma in SUBSTANTIVE_WHITELIST
                    and not re.match(r'^\d+$', lemma)):
                    tokens.append(lemma)

        tokens = apply_canonical_mapping(tokens)

    print(f"  [{region}] Total tokens: {len(tokens)}")
    compound_samples = [t for t in tokens if '_' in t][:12]
    if compound_samples:
        print(f"  [{region}] Compound samples: {compound_samples}")
    return tokens

def get_tokens_for_anchors(region):
    return get_tokens(region)

# -----------------------------
# Graph builder (HYBRID)
# -----------------------------
def _build_graph_hybrid(tokens, node_set, protected_nodes, window_size=15,
                        min_cooc=3,
                        min_npmi=0.02,
                        keep_pct=0.25,
                        protected_top_k=12,
                        protected_min_weight=0.005):
    """
    Hybrid association graph:
    - Use NPMI to select informative edges.
    - Ensure protected nodes don't "vanish": add limited top co-occurrence edges back.
    - Always include node_set nodes.
    """
    total_windows = max(len(tokens) - window_size + 1, 0)
    G = nx.Graph()
    G.add_nodes_from(node_set)

    if total_windows == 0:
        return G

    term_win = Counter()
    pair_win = defaultdict(int)

    for i in range(total_windows):
        win = [w for w in tokens[i:i+window_size] if w in node_set]
        wset = set(win)
        if not wset:
            continue
        for t in wset:
            term_win[t] += 1
        wlist = list(wset)
        for j in range(len(wlist)):
            for k in range(j+1, len(wlist)):
                u, v = sorted((wlist[j], wlist[k]))
                pair_win[(u, v)] += 1

    def npmi(c_xy, c_x, c_y, n):
        p_xy = c_xy / n
        p_x = c_x / n
        p_y = c_y / n
        if p_xy <= 0 or p_x <= 0 or p_y <= 0:
            return 0.0
        pmi = math.log(p_xy / (p_x * p_y))
        return pmi / (-math.log(p_xy))

    # 1) informative edges by NPMI
    weighted_edges = {}
    for (u, v), c_xy in pair_win.items():
        if c_xy < min_cooc:
            continue
        w = npmi(c_xy, term_win[u], term_win[v], total_windows)
        if w > min_npmi:
            weighted_edges[(u, v)] = float(w)

    # Keep top X% edges by weight for density control
    if weighted_edges and 0 < keep_pct < 1:
        vals = sorted(weighted_edges.values())
        cut = vals[int(len(vals) * (1 - keep_pct))]
        weighted_edges = {e: w for e, w in weighted_edges.items() if w >= cut}

    # 2) protected node backoff edges (by raw co-occurrence)
    protected_nodes = set(protected_nodes) & set(node_set)
    for p in protected_nodes:
        candidates = []
        for (u, v), c_xy in pair_win.items():
            if c_xy < min_cooc:
                continue
            if u == p:
                candidates.append((v, c_xy))
            elif v == p:
                candidates.append((u, c_xy))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[1], reverse=True)
        for nbr, _c_xy in candidates[:protected_top_k]:
            u, v = sorted((p, nbr))
            w = weighted_edges.get((u, v), None)
            if w is None:
                w0 = npmi(pair_win[(u, v)], term_win[u], term_win[v], total_windows)
                w = max(float(w0), protected_min_weight)
                weighted_edges[(u, v)] = w
            else:
                weighted_edges[(u, v)] = max(w, protected_min_weight)

    # 3) add edges to graph + store distance for weighted betweenness
    for (u, v), w in weighted_edges.items():
        G.add_edge(u, v, weight=float(w))
        G[u][v]['distance'] = 1.0 / (float(w) + 1e-9)

    return G

# -----------------------------
# Metrics
# -----------------------------
def _force_include_terms(top_words, counter, must_include, top_n):
    """Keep list length == top_n; force include must_include by swapping out tail items."""
    top_words = list(top_words)
    present = set(top_words)
    for t in must_include:
        if t in counter and t not in present:
            top_words.append(t)
            present.add(t)
    # trim to top_n by frequency
    top_words = sorted(top_words, key=lambda x: counter.get(x, 0), reverse=True)[:top_n]
    return top_words

def calculate_metrics(region):
    print(f"\n{'='*60}")
    print(f"Processing region: {region}")
    print(f"{'='*60}")

    tokens = get_tokens(region)
    if not tokens:
        print("  No valid tokens found.")
        return

    counter = Counter(tokens)
    top_words = [w for w, c in counter.most_common(TOP_N)]

    must_include = PROTECTED_BY_REGION.get(region, set())
    top_words = _force_include_terms(top_words, counter, must_include, TOP_N)

    node_set = set(top_words)
    protected_nodes = PROTECTED_BY_REGION.get(region, set())

    print(f"  Top 10 terms: {top_words[:10]}")

    G = _build_graph_hybrid(tokens, node_set, protected_nodes,
                            window_size=WINDOW_SIZE,
                            min_cooc=3,
                            min_npmi=0.02,
                            keep_pct=0.25,
                            protected_top_k=12,
                            protected_min_weight=0.005)

    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    degree_dict = nx.degree_centrality(G)
    betweenness_dict = nx.betweenness_centrality(G, weight='distance') if G.number_of_edges() > 0 else {n: 0.0 for n in G.nodes()}

    # eigenvector on largest CC
    if G.number_of_edges() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        Gc = G.subgraph(largest_cc).copy()
        try:
            ev_gc = nx.eigenvector_centrality(Gc, weight='weight', max_iter=3000)
            eigenvector_dict = {n: ev_gc.get(n, 0.0) for n in G.nodes()}
        except Exception:
            eigenvector_dict = {n: 0.0 for n in G.nodes()}
    else:
        eigenvector_dict = {n: 0.0 for n in G.nodes()}

    data = []
    for node in G.nodes():
        data.append({
            'Term': node,
            'Frequency': int(counter.get(node, 0)),
            'Degree_Centrality': round(float(degree_dict.get(node, 0.0)), 4),
            'Betweenness': round(float(betweenness_dict.get(node, 0.0)), 4),
            'Eigenvector': round(float(eigenvector_dict.get(node, 0.0)), 4)
        })

    df = pd.DataFrame(data).sort_values(by='Degree_Centrality', ascending=False).reset_index(drop=True)

    csv_path = os.path.join(OUTPUT_DIR, f"{region}_metrics.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Results saved: {csv_path}")

    # EU no-core robustness
    if region == 'EU' and 'ai_system' in node_set:
        tokens_no_core = [t for t in tokens if t != 'ai_system']
        if len(tokens_no_core) > WINDOW_SIZE:
            counter_nc = Counter(tokens_no_core)
            top_words_nc = [w for w, c in counter_nc.most_common(TOP_N)]
            # still force include anchors except ai_system
            must_nc = set(PROTECTED_BY_REGION.get('EU', set())) - {'ai_system'}
            top_words_nc = _force_include_terms(top_words_nc, counter_nc, must_nc, TOP_N)
            node_set_nc = set(top_words_nc)

            Gnc = _build_graph_hybrid(tokens_no_core, node_set_nc, must_nc,
                                      window_size=WINDOW_SIZE,
                                      min_cooc=3, min_npmi=0.02, keep_pct=0.25,
                                      protected_top_k=12, protected_min_weight=0.005)

            deg_nc = nx.degree_centrality(Gnc)
            bet_nc = nx.betweenness_centrality(Gnc, weight='distance') if Gnc.number_of_edges() > 0 else {n: 0.0 for n in Gnc.nodes()}
            if Gnc.number_of_edges() > 0:
                largest_cc_nc = max(nx.connected_components(Gnc), key=len)
                Gnc_cc = Gnc.subgraph(largest_cc_nc).copy()
                try:
                    ev_nc_cc = nx.eigenvector_centrality(Gnc_cc, weight='weight', max_iter=3000)
                    ev_nc = {n: ev_nc_cc.get(n, 0.0) for n in Gnc.nodes()}
                except Exception:
                    ev_nc = {n: 0.0 for n in Gnc.nodes()}
            else:
                ev_nc = {n: 0.0 for n in Gnc.nodes()}

            rows = []
            for n in Gnc.nodes():
                rows.append({
                    'Term': n,
                    'Frequency': int(counter_nc.get(n, 0)),
                    'Degree_Centrality': round(float(deg_nc.get(n, 0.0)), 4),
                    'Betweenness': round(float(bet_nc.get(n, 0.0)), 4),
                    'Eigenvector': round(float(ev_nc.get(n, 0.0)), 4)
                })
            df_nc = pd.DataFrame(rows).sort_values(by='Degree_Centrality', ascending=False).reset_index(drop=True)
            out_nc = os.path.join(OUTPUT_DIR, "EU_metrics_nocore.csv")
            df_nc.to_csv(out_nc, index=False, encoding='utf-8-sig')
            print(f"  [EU] Robustness saved: {out_nc} (ai_system removed)")

# -----------------------------
# Anchor co-occurrence profiles
# -----------------------------
ANCHORS = {
    'CN': ['安全', '人工智能', '权利', '风险'],
    'EU': ['safety', 'ai', 'right', 'risk'],
    'US': ['safety', 'ai', 'right', 'risk'],
}

def _has_component(token: str, comp: str) -> bool:
    return comp in token.split('_')

def _is_anchor_token(region: str, anchor: str, token: str) -> bool:
    if region == 'CN':
        if token == anchor:
            return True
        if anchor == '安全':
            return token.endswith('安全') or token in {'内容安全', '数据安全', '安全评估'}
        if anchor == '风险':
            return token.startswith('风险') or token.endswith('风险') or token == '风险评估'
        if anchor == '人工智能':
            return token == '人工智能'
        if anchor == '权利':
            return token == '权利'
        return False

    if anchor == 'ai':
        return token == 'ai' or _has_component(token, 'ai')
    if anchor == 'risk':
        return token == 'risk' or _has_component(token, 'risk')
    if anchor == 'right':
        return token == 'right' or _has_component(token, 'right')
    if anchor == 'safety':
        return token == 'safety' or _has_component(token, 'safety')
    return token == anchor

def calculate_anchor_cooccurrences(region, window_size=10, top_k=10):
    anchors = ANCHORS.get(region, [])
    if not anchors:
        return

    tokens = get_tokens_for_anchors(region)
    if not tokens:
        print(f"  [{region}] No tokens for anchor profiling.")
        return

    total_windows = max(len(tokens) - window_size + 1, 0)
    if total_windows == 0:
        print(f"  [{region}] Not enough tokens for window profiling.")
        return

    term_window_counts = Counter()
    anchor_window_counts = Counter()
    pair_counts = {a: Counter() for a in anchors}

    for i in range(total_windows):
        win_set = set(tokens[i:i+window_size])

        for t in win_set:
            term_window_counts[t] += 1

        for a in anchors:
            hit = {t for t in win_set if _is_anchor_token(region, a, t)}
            if not hit:
                continue
            anchor_window_counts[a] += 1
            for other in win_set - hit:
                pair_counts[a][other] += 1

    rows = []
    for a in anchors:
        a_cnt = anchor_window_counts[a]
        if a_cnt == 0:
            continue

        for other, c_xy in pair_counts[a].items():
            p_xy = c_xy / total_windows
            p_a = a_cnt / total_windows
            p_o = term_window_counts[other] / total_windows
            if p_xy > 0 and p_a > 0 and p_o > 0:
                pmi = math.log(p_xy / (p_a * p_o))
                npmi = pmi / (-math.log(p_xy))
            else:
                npmi = 0.0

            rows.append({
                "Region": region,
                "Anchor": a,
                "Neighbor": other,
                "Cooc_Windows": int(c_xy),
                "Anchor_WindowCount": int(a_cnt),
                "Neighbor_WindowCount": int(term_window_counts[other]),
                "NPMI": round(float(npmi), 4),
            })

    if not rows:
        print(f"  [{region}] No anchor co-occurrences found.")
        return

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, f"{region}_anchor_cooccurrences.csv")
    df.sort_values(["Anchor", "Cooc_Windows", "NPMI"], ascending=[True, False, False]).to_csv(
        out_path, index=False, encoding='utf-8-sig'
    )

    print(f"  [{region}] Anchor co-occurrences saved: {out_path}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AI Governance Policy Discourse Network Analysis")
    print("="*60)

    for r in ['CN', 'EU', 'US']:
        calculate_metrics(r)
        calculate_anchor_cooccurrences(r, window_size=WINDOW_SIZE, top_k=10)

    print("\n" + "="*60)
    print("  Processing complete.")
    print("="*60)
