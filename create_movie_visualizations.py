# create_movie_visualizations.py
"""
Generate visually appealing movie sentiment visualizations
for portfolio and social media images.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Create output folder
import os
os.makedirs("portfolio_images", exist_ok=True)

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12


# ============================================================
# SAMPLE MOVIE DATA
# ============================================================

MOVIES = [
    {"title": "The Shawshank Redemption", "sentiment": "Positive", "score": 0.98, "emoji": "🎬"},
    {"title": "The Godfather", "sentiment": "Positive", "score": 0.96, "emoji": "🎭"},
    {"title": "The Dark Knight", "sentiment": "Positive", "score": 0.94, "emoji": "🦇"},
    {"title": "Pulp Fiction", "sentiment": "Positive", "score": 0.91, "emoji": "🎬"},
    {"title": "Forrest Gump", "sentiment": "Positive", "score": 0.89, "emoji": "🏃"},
    {"title": "Inception", "sentiment": "Positive", "score": 0.87, "emoji": "💭"},
    {"title": "The Room", "sentiment": "Negative", "score": 0.15, "emoji": "😬"},
    {"title": "Cats (2019)", "sentiment": "Negative", "score": 0.12, "emoji": "🐱"},
    {"title": "Battlefield Earth", "sentiment": "Negative", "score": 0.08, "emoji": "👽"},
    {"title": "Disaster Movie", "sentiment": "Negative", "score": 0.05, "emoji": "💥"},
]

REVIEWS = [
    {"review": "A masterpiece of storytelling and emotion", "sentiment": "Positive", "score": 0.97},
    {"review": "Incredible performances from the entire cast", "sentiment": "Positive", "score": 0.94},
    {"review": "Visually stunning with a gripping plot", "sentiment": "Positive", "score": 0.91},
    {"review": "Waste of time, poorly written script", "sentiment": "Negative", "score": 0.08},
    {"review": "Boring and predictable from start to finish", "sentiment": "Negative", "score": 0.11},
    {"review": "Terrible acting, couldn't finish watching", "sentiment": "Negative", "score": 0.06},
]


# ============================================================
# 1. MOVIE RANKING CARDS (Best for Social Media)
# ============================================================

def create_movie_ranking_cards():
    """Create movie ranking visualization with card style."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Dark theme background
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    # Title
    ax.text(0.5, 0.95, '🎬 MOVIE SENTIMENT ANALYSIS', 
            ha='center', va='center', fontsize=24, fontweight='bold', 
            color='white', transform=ax.transAxes)
    ax.text(0.5, 0.90, 'AI-Powered Review Classification', 
            ha='center', va='center', fontsize=14, color='#8b949e', 
            transform=ax.transAxes)
    
    # Top 5 Positive
    ax.text(0.25, 0.82, '✅ TOP RATED', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='#3fb950', transform=ax.transAxes)
    
    top_movies = [m for m in MOVIES if m['sentiment'] == 'Positive'][:5]
    for i, movie in enumerate(top_movies):
        y = 0.72 - i * 0.12
        
        # Card background
        rect = mpatches.FancyBboxPatch((0.05, y - 0.04), 0.4, 0.10,
                                        boxstyle="round,pad=0.02",
                                        facecolor='#161b22', edgecolor='#3fb950',
                                        linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        
        # Rank number
        ax.text(0.08, y, f"#{i+1}", ha='center', va='center', fontsize=18, 
                fontweight='bold', color='#3fb950', transform=ax.transAxes)
        
        # Movie title
        ax.text(0.12, y + 0.01, movie['title'], ha='left', va='center', 
                fontsize=12, fontweight='bold', color='white', transform=ax.transAxes)
        
        # Score bar
        bar_width = movie['score'] * 0.25
        rect_bar = mpatches.Rectangle((0.12, y - 0.025), bar_width, 0.02,
                                       facecolor='#3fb950', alpha=0.7, transform=ax.transAxes)
        ax.add_patch(rect_bar)
        
        # Score text
        ax.text(0.42, y, f"{movie['score']*100:.0f}%", ha='right', va='center', 
                fontsize=12, fontweight='bold', color='#3fb950', transform=ax.transAxes)
    
    # Bottom 5 Negative
    ax.text(0.75, 0.82, '❌ LOWEST RATED', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='#f85149', transform=ax.transAxes)
    
    bottom_movies = [m for m in MOVIES if m['sentiment'] == 'Negative'][:5]
    for i, movie in enumerate(bottom_movies):
        y = 0.72 - i * 0.12
        
        # Card background
        rect = mpatches.FancyBboxPatch((0.55, y - 0.04), 0.4, 0.10,
                                        boxstyle="round,pad=0.02",
                                        facecolor='#161b22', edgecolor='#f85149',
                                        linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        
        # Rank number
        ax.text(0.58, y, f"#{i+1}", ha='center', va='center', fontsize=18, 
                fontweight='bold', color='#f85149', transform=ax.transAxes)
        
        # Movie title
        ax.text(0.62, y + 0.01, movie['title'], ha='left', va='center', 
                fontsize=12, fontweight='bold', color='white', transform=ax.transAxes)
        
        # Score bar
        bar_width = movie['score'] * 0.25
        rect_bar = mpatches.Rectangle((0.62, y - 0.025), bar_width, 0.02,
                                       facecolor='#f85149', alpha=0.7, transform=ax.transAxes)
        ax.add_patch(rect_bar)
        
        # Score text
        ax.text(0.92, y, f"{movie['score']*100:.0f}%", ha='right', va='center', 
                fontsize=12, fontweight='bold', color='#f85149', transform=ax.transAxes)
    
    # Footer
    ax.text(0.5, 0.05, 'Powered by DistilBERT + PyTorch + AWS S3', 
            ha='center', va='center', fontsize=10, color='#8b949e', transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.savefig('portfolio_images/01_movie_ranking_cards.png', dpi=300, 
                bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print("✅ Created: portfolio_images/01_movie_ranking_cards.png")


# ============================================================
# 2. SENTIMENT METER DASHBOARD
# ============================================================

def create_sentiment_dashboard():
    """Create a dashboard-style sentiment visualization."""
    
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Main title
    fig.text(0.5, 0.95, '🎬 SENTIMENT ANALYSIS DASHBOARD', 
             ha='center', fontsize=28, fontweight='bold', color='white')
    fig.text(0.5, 0.91, 'Real-time Movie Review Classification', 
             ha='center', fontsize=14, color='#888888')
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3, 
                          left=0.08, right=0.92, top=0.85, bottom=0.1)
    
    # ---- Panel 1: Gauge Chart ----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#16213e')
    
    # Draw gauge
    theta = np.linspace(0, np.pi, 100)
    r_outer = 1
    r_inner = 0.6
    
    # Background arc
    ax1.fill_between(theta, r_inner, r_outer, alpha=0.3, color='#444')
    
    # Colored sections
    ax1.fill_between(theta[:33], r_inner, r_outer, color='#f85149', alpha=0.8)
    ax1.fill_between(theta[33:66], r_inner, r_outer, color='#f0ad4e', alpha=0.8)
    ax1.fill_between(theta[66:], r_inner, r_outer, color='#3fb950', alpha=0.8)
    
    # Needle (85% = 0.85 * pi)
    needle_angle = 0.85 * np.pi
    ax1.annotate('', xy=(needle_angle, 0.95), xytext=(np.pi/2, 0),
                 arrowprops=dict(arrowstyle='->', color='white', lw=3),
                 transform=ax1.transData + plt.matplotlib.transforms.Affine2D().scale(1, 1))
    
    ax1.plot([np.pi/2], [0], 'wo', markersize=15)
    ax1.text(np.pi/2, -0.3, '85%', ha='center', fontsize=24, fontweight='bold', color='#3fb950')
    ax1.text(np.pi/2, -0.5, 'ACCURACY', ha='center', fontsize=12, color='white')
    
    ax1.set_xlim(0, np.pi)
    ax1.set_ylim(-0.6, 1.1)
    ax1.axis('off')
    ax1.set_title('Model Performance', color='white', fontsize=14, pad=10)
    
    # ---- Panel 2: Live Predictions ----
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_facecolor('#16213e')
    
    reviews_display = REVIEWS[:4]
    for i, review in enumerate(reviews_display):
        y = 0.85 - i * 0.22
        color = '#3fb950' if review['sentiment'] == 'Positive' else '#f85149'
        symbol = '✅' if review['sentiment'] == 'Positive' else '❌'
        
        # Review text
        ax2.text(0.02, y, f'"{review["review"][:45]}..."', 
                 ha='left', va='center', fontsize=10, color='white', transform=ax2.transAxes)
        
        # Result
        ax2.text(0.85, y, f'{symbol} {review["score"]*100:.0f}%', 
                 ha='center', va='center', fontsize=12, fontweight='bold', 
                 color=color, transform=ax2.transAxes)
    
    ax2.axis('off')
    ax2.set_title('Live Sentiment Predictions', color='white', fontsize=14, pad=10)
    
    # ---- Panel 3: Distribution Pie ----
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#16213e')
    
    sizes = [5331, 5331]
    colors = ['#3fb950', '#f85149']
    explode = (0.02, 0.02)
    
    wedges, texts, autotexts = ax3.pie(sizes, explode=explode, colors=colors,
                                        autopct='%1.0f%%', startangle=90,
                                        textprops={'color': 'white', 'fontsize': 14})
    ax3.legend(['Positive', 'Negative'], loc='lower center', 
               fontsize=10, facecolor='#16213e', edgecolor='none', labelcolor='white')
    ax3.set_title('Dataset Balance', color='white', fontsize=14, pad=10)
    
    # ---- Panel 4: Top Movies Bar ----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#16213e')
    
    top_5 = [m for m in MOVIES if m['sentiment'] == 'Positive'][:5]
    titles = [m['title'][:15] + '...' if len(m['title']) > 15 else m['title'] for m in top_5]
    scores = [m['score'] * 100 for m in top_5]
    
    bars = ax4.barh(titles, scores, color='#3fb950', edgecolor='white', linewidth=1)
    ax4.set_xlim(0, 105)
    ax4.invert_yaxis()
    ax4.set_xlabel('Sentiment Score (%)', color='white')
    ax4.tick_params(colors='white')
    ax4.set_title('Top Rated Movies', color='white', fontsize=14, pad=10)
    
    for spine in ax4.spines.values():
        spine.set_color('#444')
    
    # ---- Panel 5: Stats Cards ----
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor('#16213e')
    
    stats = [
        {'label': 'Reviews', 'value': '10,662', 'color': '#58a6ff'},
        {'label': 'Precision', 'value': '86%', 'color': '#3fb950'},
        {'label': 'Recall', 'value': '84%', 'color': '#f0ad4e'},
        {'label': 'F1 Score', 'value': '85%', 'color': '#f85149'},
    ]
    
    for i, stat in enumerate(stats):
        y = 0.82 - i * 0.22
        ax5.text(0.5, y, stat['value'], ha='center', va='center', 
                 fontsize=28, fontweight='bold', color=stat['color'], transform=ax5.transAxes)
        ax5.text(0.5, y - 0.08, stat['label'], ha='center', va='center', 
                 fontsize=12, color='white', transform=ax5.transAxes)
    
    ax5.axis('off')
    ax5.set_title('Key Metrics', color='white', fontsize=14, pad=10)
    
    plt.savefig('portfolio_images/02_sentiment_dashboard.png', dpi=300, 
                bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("✅ Created: portfolio_images/02_sentiment_dashboard.png")


# ============================================================
# 3. MOVIE RATING COMPARISON (Instagram Style)
# ============================================================

def create_instagram_style():
    """Create Instagram-friendly square visualization."""
    
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    # Title
    ax.text(0.5, 0.94, '🎬 AI MOVIE RATINGS', ha='center', va='center',
            fontsize=28, fontweight='bold', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.89, 'Sentiment Analysis Results', ha='center', va='center',
            fontsize=14, color='#8b949e', transform=ax.transAxes)
    
    # Divider
    ax.plot([0.1, 0.9], [0.85, 0.85], color='#30363d', linewidth=2, transform=ax.transAxes)
    
    # Movies list
    all_movies = sorted(MOVIES, key=lambda x: x['score'], reverse=True)
    
    for i, movie in enumerate(all_movies):
        y = 0.78 - i * 0.075
        
        # Rank
        rank_color = '#3fb950' if i < 5 else '#f85149'
        ax.text(0.08, y, f'{i+1}', ha='center', va='center', fontsize=16, 
                fontweight='bold', color=rank_color, transform=ax.transAxes)
        
        # Movie title
        ax.text(0.14, y, movie['title'], ha='left', va='center', fontsize=12, 
                color='white', transform=ax.transAxes)
        
        # Score bar background
        rect_bg = mpatches.Rectangle((0.55, y - 0.015), 0.35, 0.03,
                                      facecolor='#21262d', transform=ax.transAxes)
        ax.add_patch(rect_bg)
        
        # Score bar fill
        bar_color = '#3fb950' if movie['sentiment'] == 'Positive' else '#f85149'
        bar_width = movie['score'] * 0.35
        rect_fill = mpatches.Rectangle((0.55, y - 0.015), bar_width, 0.03,
                                        facecolor=bar_color, transform=ax.transAxes)
        ax.add_patch(rect_fill)
        
        # Score text
        ax.text(0.92, y, f"{movie['score']*100:.0f}%", ha='right', va='center',
                fontsize=11, fontweight='bold', color=bar_color, transform=ax.transAxes)
    
    # Footer stats
    ax.plot([0.1, 0.9], [0.05, 0.05], color='#30363d', linewidth=2, transform=ax.transAxes)
    
    footer_stats = [
        ('🎯 85%', 'Accuracy'),
        ('📊 10,662', 'Reviews'),
        ('🤖 DistilBERT', 'Model'),
    ]
    
    for i, (value, label) in enumerate(footer_stats):
        x = 0.22 + i * 0.28
        ax.text(x, 0.025, value, ha='center', va='center', fontsize=12, 
                fontweight='bold', color='white', transform=ax.transAxes)
        ax.text(x, 0.005, label, ha='center', va='center', fontsize=9, 
                color='#8b949e', transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.savefig('portfolio_images/03_instagram_style.png', dpi=300, 
                bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print("✅ Created: portfolio_images/03_instagram_style.png")


# ============================================================
# 4. NEON STYLE VISUALIZATION
# ============================================================

def create_neon_style():
    """Create neon/cyberpunk style visualization."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Neon title with glow effect
    ax.text(0.5, 0.92, 'SENTIMENT', ha='center', va='center',
            fontsize=48, fontweight='bold', color='#00ffff', 
            transform=ax.transAxes, alpha=0.9)
    ax.text(0.5, 0.84, 'ANALYSIS', ha='center', va='center',
            fontsize=48, fontweight='bold', color='#ff00ff', 
            transform=ax.transAxes, alpha=0.9)
    
    # Subtitle
    ax.text(0.5, 0.76, '[ NEURAL NETWORK POWERED ]', ha='center', va='center',
            fontsize=14, color='#00ff00', transform=ax.transAxes,
            family='monospace')
    
    # Stats boxes
    stats = [
        {'value': '85%', 'label': 'ACCURACY', 'color': '#00ffff', 'x': 0.2},
        {'value': '10.6K', 'label': 'REVIEWS', 'color': '#ff00ff', 'x': 0.4},
        {'value': '86%', 'label': 'PRECISION', 'color': '#00ff00', 'x': 0.6},
        {'value': '84%', 'label': 'RECALL', 'color': '#ffff00', 'x': 0.8},
    ]
    
    for stat in stats:
        # Glowing box
        rect = mpatches.FancyBboxPatch(
            (stat['x'] - 0.08, 0.48), 0.16, 0.22,
            boxstyle="round,pad=0.02",
            facecolor='#0a0a0a',
            edgecolor=stat['color'],
            linewidth=3,
            transform=ax.transAxes
        )
        ax.add_patch(rect)
        
        ax.text(stat['x'], 0.62, stat['value'], ha='center', va='center',
                fontsize=32, fontweight='bold', color=stat['color'],
                transform=ax.transAxes, family='monospace')
        ax.text(stat['x'], 0.52, stat['label'], ha='center', va='center',
                fontsize=10, color='#666666', transform=ax.transAxes,
                family='monospace')
    
    # Bottom section - sample predictions
    ax.text(0.5, 0.38, '─── SAMPLE PREDICTIONS ───', ha='center', va='center',
            fontsize=12, color='#444444', transform=ax.transAxes, family='monospace')
    
    samples = [
        ('"Amazing movie, must watch!"', 'POSITIVE', '#00ff00', '97%'),
        ('"Terrible waste of time"', 'NEGATIVE', '#ff0000', '94%'),
        ('"Great acting and story"', 'POSITIVE', '#00ff00', '91%'),
    ]
    
    for i, (review, sentiment, color, score) in enumerate(samples):
        y = 0.28 - i * 0.08
        ax.text(0.15, y, review, ha='left', va='center', fontsize=11, 
                color='#888888', transform=ax.transAxes, family='monospace')
        ax.text(0.70, y, sentiment, ha='center', va='center', fontsize=11, 
                fontweight='bold', color=color, transform=ax.transAxes, family='monospace')
        ax.text(0.85, y, score, ha='center', va='center', fontsize=11, 
                color=color, transform=ax.transAxes, family='monospace')
    
    # Tech stack footer
    ax.text(0.5, 0.05, 'PyTorch  ●  DistilBERT  ●  AWS S3  ●  Streamlit', 
            ha='center', va='center', fontsize=10, color='#444444', 
            transform=ax.transAxes, family='monospace')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.savefig('portfolio_images/04_neon_style.png', dpi=300, 
                bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print("✅ Created: portfolio_images/04_neon_style.png")


# ============================================================
# 5. CLEAN MINIMAL STYLE
# ============================================================

def create_minimal_style():
    """Create clean, minimal professional style."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    # Title
    ax.text(0.5, 0.92, 'Sentiment Analysis Results', ha='center', va='center',
            fontsize=28, fontweight='bold', color='#1a1a1a', transform=ax.transAxes)
    ax.text(0.5, 0.86, 'Movie Review Classification using Deep Learning', ha='center', va='center',
            fontsize=12, color='#666666', transform=ax.transAxes)
    
    # Metrics row
    metrics = [
        {'value': '85%', 'label': 'Accuracy', 'x': 0.2},
        {'value': '86%', 'label': 'Precision', 'x': 0.4},
        {'value': '84%', 'label': 'Recall', 'x': 0.6},
        {'value': '85%', 'label': 'F1 Score', 'x': 0.8},
    ]
    
    for metric in metrics:
        ax.text(metric['x'], 0.72, metric['value'], ha='center', va='center',
                fontsize=36, fontweight='bold', color='#2563eb', transform=ax.transAxes)
        ax.text(metric['x'], 0.64, metric['label'], ha='center', va='center',
                fontsize=11, color='#666666', transform=ax.transAxes)
    
    # Divider
    ax.plot([0.1, 0.9], [0.56, 0.56], color='#e5e5e5', linewidth=2, transform=ax.transAxes)
    
    # Top predictions
    ax.text(0.5, 0.50, 'Sample Classifications', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#1a1a1a', transform=ax.transAxes)
    
    predictions = [
        ('The Shawshank Redemption', 'Positive', 98),
        ('The Godfather', 'Positive', 96),
        ('The Dark Knight', 'Positive', 94),
        ('Cats (2019)', 'Negative', 12),
        ('Disaster Movie', 'Negative', 5),
    ]
    
    for i, (title, sentiment, score) in enumerate(predictions):
        y = 0.42 - i * 0.07
        color = '#22c55e' if sentiment == 'Positive' else '#ef4444'
        
        ax.text(0.15, y, title, ha='left', va='center', fontsize=11, 
                color='#1a1a1a', transform=ax.transAxes)
        
        # Progress bar
        rect_bg = mpatches.Rectangle((0.50, y - 0.012), 0.30, 0.024,
                                      facecolor='#f3f4f6', transform=ax.transAxes)
        ax.add_patch(rect_bg)
        
        rect_fill = mpatches.Rectangle((0.50, y - 0.012), score/100 * 0.30, 0.024,
                                        facecolor=color, transform=ax.transAxes)
        ax.add_patch(rect_fill)
        
        ax.text(0.85, y, f'{score}%', ha='center', va='center', fontsize=11, 
                fontweight='bold', color=color, transform=ax.transAxes)
    
    # Footer
    ax.text(0.5, 0.06, 'Built with PyTorch • Hugging Face • AWS S3 • Streamlit', 
            ha='center', va='center', fontsize=10, color='#999999', transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.savefig('portfolio_images/05_minimal_style.png', dpi=300, 
                bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print("✅ Created: portfolio_images/05_minimal_style.png")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🎨 GENERATING PORTFOLIO VISUALIZATIONS")
    print("=" * 50 + "\n")
    
    create_movie_ranking_cards()
    create_sentiment_dashboard()
    create_instagram_style()
    create_neon_style()
    create_minimal_style()
    
    print("\n" + "=" * 50)
    print("✅ ALL IMAGES CREATED!")
    print("📁 Check the 'portfolio_images' folder")
    print("=" * 50)
    print("\nFiles created:")
    print("  1. 01_movie_ranking_cards.png  - Dark theme with rankings")
    print("  2. 02_sentiment_dashboard.png  - Full dashboard view")
    print("  3. 03_instagram_style.png      - Square format for social")
    print("  4. 04_neon_style.png           - Cyberpunk/neon aesthetic")
    print("  5. 05_minimal_style.png        - Clean professional look")
    print("\n")