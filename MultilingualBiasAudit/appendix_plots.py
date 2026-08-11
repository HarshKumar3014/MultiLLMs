"""
appendix_plots.py — Per-category CLFI figure (Appendix)

Generates fig6_category_clfi: mean CLFI computed independently per bias
category. Descriptive only; see the noise-floor result in the paper before
reading any per-category ordering as meaningful.

Usage:
    python appendix_plots.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR, FIGURES_DIR

df = pd.read_csv(RESULTS_DIR / 'all_results.csv')

top_cats = df.groupby('category')['prompt_id'].count().nlargest(10).index
models = sorted(df['model'].unique())

cat_means = []
cat_stds = []
for cat in top_cats:
    cdf = df[df['category'] == cat]
    cat_clfis = []
    for m in models:
        mdf = cdf[cdf['model']==m]
        if len(mdf) == 0: continue
        en_m = mdf[mdf['language']=='en']['stereotype_score'].mean()
        dfgs = []
        for lang in mdf['language'].unique():
            if lang == 'en': continue
            ldf = mdf[mdf['language']==lang]
            if len(ldf) == 0: continue
            dfgs.append(abs(ldf['stereotype_score'].mean() - en_m))
        if dfgs:
            cat_clfis.append(1 - np.mean(dfgs) / 0.5)
    cat_means.append(np.mean(cat_clfis))
    cat_stds.append(np.std(cat_clfis))

# Change figsize from (10,6) to (10, 3.5) to make it shorter and wider
plt.figure(figsize=(10, 3.5))
sns.barplot(x=cat_means, y=top_cats, orient='h', palette='viridis', errorbar=None)
plt.errorbar(x=cat_means, y=range(len(top_cats)), xerr=cat_stds, fmt='none', c='black', capsize=5)
plt.xlim(0.8, 1.0)
plt.xlabel('Mean Cross-Lingual Fairness Index (CLFI)')
plt.ylabel('Bias Category')
plt.title('CLFI by Bias Category (with Std Dev)')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig6_category_clfi.pdf')
plt.savefig(FIGURES_DIR / 'fig6_category_clfi.png', dpi=300)
plt.close()
