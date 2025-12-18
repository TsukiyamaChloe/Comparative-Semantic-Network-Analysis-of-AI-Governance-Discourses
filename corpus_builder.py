import os
import re
import pdfplumber

BASE_DIR = os.getcwd()
RAW_DIR = os.path.join(BASE_DIR, 'raw_pdfs')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_txt')

EN_BROKEN_FIXES = [
    (r'ar tificial', 'artificial'), (r'Ar tificial', 'Artificial'),
    (r'ar tif icial', 'artificial'), (r'Ar tif icial', 'Artificial'),
    (r'inte lligence', 'intelligence'), (r'intellig ence', 'intelligence'),
    (r'Intellig ence', 'Intelligence'),
    (r'sys tem', 'system'), (r'syste m', 'system'), (r'Syste m', 'System'),
    (r'syste ms', 'systems'),
    (r'r isk', 'risk'), (r'R isk', 'Risk'),
    (r'har m', 'harm'), (r'Har m', 'Harm'),
    (r'saf ety', 'safety'), (r'Saf ety', 'Safety'),
    (r'secur ity', 'security'), (r'Secur ity', 'Security'),
    (r'biometr ic', 'biometric'), (r'Biometr ic', 'Biometric'),
    (r'cybersecur ity', 'cybersecurity'),
    (r'g eneral', 'general'), (r'G eneral', 'General'),
    (r'inf or mation', 'information'), (r'infor mation', 'information'),
    (r'perf or mance', 'performance'), (r'perfor mance', 'performance'),
    (r'conf or mity', 'conformity'), (r'confor mity', 'conformity'),
    (r'regulat or y', 'regulatory'), (r'regulator y', 'regulatory'),
    (r'classif ied', 'classified'), (r'classif ication', 'classification'),
    (r'specif ic', 'specific'), (r'specif ied', 'specified'),
    (r'transpar ency', 'transparency'), (r'T ranspar ency', 'Transparency'),
    (r'fundament al', 'fundamental'), (r'Fundament al', 'Fundamental'),
    (r'requir ements', 'requirements'), (r'requir ing', 'requiring'),
    (r'tec hnical', 'technical'), (r'te chnical', 'technical'), (r'T ec hnical', 'Technical'),
    (r'im plementation', 'implementation'), (r'imp lementation', 'implementation'),
    (r'cooper ation', 'cooperation'), (r'operat ion', 'operation'),
    (r'provi der', 'provider'), (r'Provi der', 'Provider'),
    (r'deplo y er', 'deployer'), (r'Deplo y er', 'Deployer'), (r'deployer s', 'deployers'),
    (r'imp or ter', 'importer'), (r'Imp or ter', 'Importer'),
    (r'distr ibutor', 'distributor'), (r'Distr ibutor', 'Distributor'),
    (r'operat or', 'operator'), (r'Operat or', 'Operator'),
    (r'aut hor ity', 'authority'), (r'Aut hor ity', 'Authority'), (r'author ities', 'authorities'),
    (r'committ ee', 'committee'), (r'Committ ee', 'Committee'),
    (r'exper ts', 'experts'), (r'Exper ts', 'Experts'),
    (r' f or ', ' for '), (r' f rom ', ' from '), (r' t o ', ' to '), (r' o f ', ' of '),
]

CN_FIXES = [
    (r'人人工', '人机'), 
    (r'智能动', '“人工智能+”行动'),
    (r'新质产', '新质生产力'),
    (r'民福祉', '人民福祉'),
    (r'智能经济', '人工智能经济'),
    (r'智能社会', '人工智能社会'),
    (r'智能发展', '人工智能发展'),
    (r'业全要素', '工业全要素'),
    (r'造福类', '造福人类'),
    (r'算、数据', '算力、数据'),
    (r'智能互联', '人工智能互联'),
    (r'平开放', '水平开放'),
]

def extract_text_with_cropping(pdf, region):
    full_text = []
    
    for page in pdf.pages:
        width = page.width
        height = page.height

        if region == 'EU':
            top_margin, bottom_margin = 70, 70
        elif region == 'CN':
            top_margin, bottom_margin = 60, 50
        elif region == 'US':
            top_margin, bottom_margin = 50, 50
        else:
            top_margin, bottom_margin = 50, 50
            
        crop_box = (0, top_margin, width, height - bottom_margin)
        
        try:
            cropped_page = page.crop(crop_box)
            
            if region == 'US':
                text = cropped_page.extract_text(x_tolerance=1)
            else:
                text = cropped_page.extract_text()
                
            if text:
                full_text.append(text)
        except ValueError:
            text = page.extract_text(x_tolerance=1 if region == 'US' else 3)
            if text:
                full_text.append(text)
                
    return "\n".join(full_text)

def clean_text_content(text, region):
    if not text: return ""

    text = re.sub(r'[\x00-\x1f]', '', text)

    text = text.replace('\n', ' ')

    if region in ['EU', 'US']:

        text = re.sub(r'(\w+)\s+-\s+(\w+)', r'\1\2', text)

        text = re.sub(r'([.,;:!?])([a-zA-Z])', r'\1 \2', text)

        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        text = re.sub(r'https?://\S+', '', text)

        text = re.sub(r'Page\s*\d+', '', text, flags=re.IGNORECASE)

        for pattern, replacement in EN_BROKEN_FIXES:
            text = re.sub(pattern, replacement, text)

        def fix_wide_caps(match):
            return match.group(0).replace(' ', '')
        text = re.sub(r'\b([A-Z] ){3,}[A-Z]\b', fix_wide_caps, text)

    elif region == 'CN':
        for pattern, replacement in CN_FIXES:
            text = text.replace(pattern, replacement)

        text = re.sub(r'\s+', '', text)
        text = re.sub(r'www\.gov\.cn', '', text)


    if region != 'CN':
        text = re.sub(r'\s+', ' ', text).strip()
        
    return text

def process_pdfs():
    regions = ['CN', 'EU', 'US']
    
    for region in regions:
        input_path = os.path.join(RAW_DIR, region)
        output_path = os.path.join(PROCESSED_DIR, region)
        os.makedirs(output_path, exist_ok=True)
        
        full_region_text = ""
        
        if not os.path.exists(input_path):
            continue
            
        files = sorted([f for f in os.listdir(input_path) if f.lower().endswith('.pdf')])
        print(f"Processing Region: {region} ({len(files)} files)")
        
        for filename in files:
            file_path = os.path.join(input_path, filename)
            try:
                with pdfplumber.open(file_path) as pdf:
                    raw_text = extract_text_with_cropping(pdf, region)
                    clean_content = clean_text_content(raw_text, region)

                txt_filename = filename.replace('.pdf', '.txt')
                with open(os.path.join(output_path, txt_filename), 'w', encoding='utf-8') as f:
                    f.write(clean_content)

                full_region_text += f"\n\n--- DOCUMENT: {filename} ---\n\n" + clean_content

                print(f" Converted: {filename} | Length: {len(clean_content)} chars")
                
            except Exception as e:
                print(f" Error processing {filename}: {str(e)}")
        
        merged_filename = os.path.join(PROCESSED_DIR, f"{region}_ALL.txt")
        with open(merged_filename, 'w', encoding='utf-8') as f:
            f.write(full_region_text)
        print(f"  MERGED SAVED: {merged_filename} (Total Length: {len(full_region_text)})")

if __name__ == "__main__":
    process_pdfs()