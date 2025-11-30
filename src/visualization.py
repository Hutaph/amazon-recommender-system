import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'plot_mf_interaction_density',
    'plot_learning_curve',
    'plot_baseline_scatter',
    'plot_rating_distribution',
    'plot_user_activity',
    'plot_product_popularity',
    'plot_temporal_patterns',
    'plot_long_tail',
    'plot_sparsity_sample',
    'plot_top_users',
    'plot_top_products_avg_rating',
    'plot_statistics_summary',
    'plot_kcore_impact_comparison',
    'plot_rating_distribution_comparison',
    'plot_activity_comparison',
    'plot_interaction_lorenz',
    'plot_kcore_tradeoff_curve',
    'plot_timestamp_integrity',
    'plot_bias_violin',
    'plot_train_test_distribution',
]

def plot_mf_interaction_density(user_counts, item_counts, figsize=(14, 5)):
    if user_counts.size == 0 or item_counts.size == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].hist(user_counts, bins=50, edgecolor='black', alpha=0.75, color='steelblue')
    axes[0].axvline(np.mean(user_counts), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(user_counts):.2f}")
    axes[0].axvline(np.median(user_counts), color='green', linestyle='--', linewidth=2, label=f"Median: {np.median(user_counts):.1f}")
    axes[0].set_xlabel('So luong tuong tac moi user')
    axes[0].set_ylabel('So luong user')
    axes[0].set_title('Phan bo tuong tac per User')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(item_counts, bins=50, edgecolor='black', alpha=0.75, color='indianred')
    axes[1].axvline(np.mean(item_counts), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(item_counts):.2f}")
    axes[1].axvline(np.median(item_counts), color='green', linestyle='--', linewidth=2, label=f"Median: {np.median(item_counts):.1f}")
    axes[1].set_xlabel('So luong tuong tac moi san pham')
    axes[1].set_ylabel('So luong san pham')
    axes[1].set_title('Phan bo tuong tac per Item')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.suptitle('Mat do tin hieu danh cho Matrix Factorization', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    print('Nhan xet nhanh:')
    print('- Histogram user cho thay median < mean -> long-tail da duoc cat duoi bang k-core.')
    print('- Histogram item rong hon, can gioi han latent factors de tranh hoc noise tu item hiem.')
    print('- Moi ben van co >=5 tuong tac o phan lon thuc the -> du tin hieu de build MF.')

def plot_learning_curve(train_history, test_history=None, title='MF learning curve', figsize=(6, 4)):
    if not train_history and not test_history:
        return
    fig, ax = plt.subplots(figsize=figsize)
    if train_history:
        ax.plot(train_history, label='Train RMSE', color='#4e79a7')
    if test_history:
        ax.plot(test_history, label='Test RMSE', color='#f28e2b')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_baseline_scatter(test_matrix, model, sample_size=1000, figsize=(6, 4)):
    if test_matrix.size == 0:
        return
    n = test_matrix.shape[0]
    idx = np.random.choice(n, size=min(sample_size, n), replace=False)
    truth = test_matrix[idx, 2]
    preds = model.predict_vector(test_matrix[idx, 1].astype(int))
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(truth, preds, alpha=0.4, s=15)
    ax.set_title('Baseline predictions vs truth (sample)')
    ax.set_xlabel('True rating (z-score)')
    ax.set_ylabel('Predicted rating')
    ax.axline((0, 0), slope=1, color='red', linestyle='--', alpha=0.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_rating_distribution(ratings, title='Rating Distribution', figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    bins = np.arange(0.5, 6, 0.5)
    counts, edges = np.histogram(ratings, bins=bins)
    ax.bar(edges[:-1] + 0.25, counts, width=0.4, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(1, 6))
    ax.grid(axis='y', alpha=0.3)
    for edge, count in zip(edges[:-1], counts):
        if count > 0:
            ax.text(edge + 0.25, count, str(int(count)), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_user_activity(user_counts, title='User Activity Distribution', figsize=(12, 6), bins=50, log_scale=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.hist(user_counts, bins=bins, alpha=0.7, color='coral', edgecolor='black')
    ax1.set_xlabel('Number of Ratings per User', fontsize=11)
    ax1.set_ylabel('Number of Users', fontsize=11)
    ax1.set_title(f'{title} - Histogram', fontsize=12, fontweight='bold')
    if log_scale:
        ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3)
    stats_text = f"""
    Total Users: {len(user_counts):,}
    Mean: {np.mean(user_counts):.2f}
    Median: {np.median(user_counts):.1f}
    Std: {np.std(user_counts):.2f}
    Min: {np.min(user_counts)}
    Max: {np.max(user_counts)}
    25%: {np.percentile(user_counts, 25):.1f}
    75%: {np.percentile(user_counts, 75):.1f}
    """
    sorted_counts = np.sort(user_counts)
    cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    ax2.plot(sorted_counts, cdf, color='coral', linewidth=2)
    ax2.set_xlabel('Number of Ratings per User', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title(f'{title} - CDF', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.text(0.98, 0.02, stats_text.strip(), transform=ax2.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.show()

def plot_product_popularity(product_counts, title='Product Popularity Distribution', figsize=(12, 6), bins=50, log_scale=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.hist(product_counts, bins=bins, alpha=0.7, color='mediumseagreen', edgecolor='black')
    ax1.set_xlabel('Number of Ratings per Product', fontsize=11)
    ax1.set_ylabel('Number of Products', fontsize=11)
    ax1.set_title(f'{title} - Histogram', fontsize=12, fontweight='bold')
    if log_scale:
        ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3)
    stats_text = f"""
    Total Products: {len(product_counts):,}
    Mean: {np.mean(product_counts):.2f}
    Median: {np.median(product_counts):.1f}
    Std: {np.std(product_counts):.2f}
    Min: {np.min(product_counts)}
    Max: {np.max(product_counts)}
    25%: {np.percentile(product_counts, 25):.1f}
    75%: {np.percentile(product_counts, 75):.1f}
    """
    sorted_counts = np.sort(product_counts)[::-1]
    top_n = min(100, len(sorted_counts))
    ax2.plot(np.arange(1, top_n + 1), sorted_counts[:top_n], color='mediumseagreen', linewidth=2)
    ax2.set_xlabel('Product Rank', fontsize=11)
    ax2.set_ylabel('Number of Ratings', fontsize=11)
    ax2.set_title(f'Top {top_n} Products (Long-tail)', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(alpha=0.3)
    ax2.text(0.98, 0.98, stats_text.strip(), transform=ax2.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    plt.tight_layout()
    plt.show()

def plot_temporal_patterns(timestamps, title='Temporal Rating Patterns', figsize=(14, 6)):
    if timestamps.dtype != 'datetime64[s]':
        timestamps = timestamps.astype('int64').astype('datetime64[s]')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    years = timestamps.astype('datetime64[Y]').astype(int) + 1970
    months = timestamps.astype('datetime64[M]').astype(int) % 12 + 1
    unique_years, year_counts = np.unique(years, return_counts=True)
    ax1.bar(unique_years, year_counts, alpha=0.7, color='mediumpurple', edgecolor='black')
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Number of Ratings', fontsize=11)
    ax1.set_title('Ratings per Year', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    month_counts = np.bincount(months, minlength=13)[1:]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax2.bar(month_names, month_counts, alpha=0.7, color='mediumpurple', edgecolor='black')
    ax2.set_xlabel('Month', fontsize=11)
    ax2.set_ylabel('Number of Ratings', fontsize=11)
    ax2.set_title('Ratings per Month (All Years)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

def plot_long_tail(counts, title='Long-tail Distribution', figsize=(12, 5), top_n=100):
    sorted_counts = np.sort(counts)[::-1]
    cumsum = np.cumsum(sorted_counts)
    total = cumsum[-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.loglog(np.arange(1, len(sorted_counts) + 1), sorted_counts, color='darkviolet', linewidth=2)
    ax1.axvline(x=top_n, color='red', linestyle='--', linewidth=1.5, label=f'Top {top_n}')
    ax1.set_xlabel('Rank (log scale)', fontsize=11)
    ax1.set_ylabel('Count (log scale)', fontsize=11)
    ax1.set_title(f'{title} - Log-Log Plot', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    cumsum_pct = cumsum / total * 100
    ax2.plot(np.arange(1, len(cumsum_pct) + 1), cumsum_pct, color='darkviolet', linewidth=2)
    if len(cumsum_pct) >= top_n:
        top_n_pct = cumsum_pct[top_n - 1]
        ax2.axvline(x=top_n, color='red', linestyle='--', linewidth=1.5)
        ax2.axhline(y=top_n_pct, color='red', linestyle='--', linewidth=1.5, label=f'Top {top_n}: {top_n_pct:.1f}%')
    ax2.set_xlabel('Number of Items', fontsize=11)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=11)
    ax2.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_sparsity_sample(user_indices, product_indices, n_users=100, n_products=100, title='Interaction Matrix Sample', figsize=(10, 8)):
    max_user = user_indices.max() + 1
    max_product = product_indices.max() + 1
    sample_users = np.random.choice(max_user, size=min(n_users, max_user), replace=False)
    sample_products = np.random.choice(max_product, size=min(n_products, max_product), replace=False)
    matrix = np.zeros((len(sample_users), len(sample_products)))
    user_map = {u: i for i, u in enumerate(sample_users)}
    product_map = {p: i for i, p in enumerate(sample_products)}
    for u, p in zip(user_indices, product_indices):
        if u in user_map and p in product_map:
            matrix[user_map[u], product_map[p]] = 1
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', interpolation='nearest')
    ax.set_xlabel(f'Products (sample of {len(sample_products)})', fontsize=11)
    ax.set_ylabel(f'Users (sample of {len(sample_users)})', fontsize=11)
    ax.set_title(f'{title}\nSparsity: {100 * (1 - matrix.sum() / matrix.size):.2f}%', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Interaction')
    plt.tight_layout()
    plt.show()

def plot_top_users(user_ids, user_counts, top_n=10, title='Top 10 Khách hàng Đánh giá Nhiều nhất', figsize=(10, 6)):
    top_idx = np.argsort(user_counts)[-top_n:][::-1]
    top_users = user_ids[top_idx]
    top_counts = user_counts[top_idx]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(top_users, top_counts, color='darkred', edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 5, int(yval), ha='center', va='bottom', fontsize=10)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('ID Khách hàng', fontsize=14)
    ax.set_ylabel('Tổng số lần Đánh giá', fontsize=14)
    ax.set_ylim(0, max(top_counts) * 1.1)
    ax.tick_params(axis='x', rotation=15)
    plt.setp(ax.get_xticklabels(), ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_top_products_avg_rating(product_ids, product_ratings, top_n=10, title='Trung bình Rating của 10 Sản phẩm được đánh giá nhiều nhất', figsize=(12, 6)):
    unique_prod, inverse_idx = np.unique(product_ids, return_inverse=True)
    counts = np.bincount(inverse_idx)
    sum_ratings = np.bincount(inverse_idx, weights=product_ratings)
    avg_ratings = sum_ratings / counts
    top_idx = np.argsort(counts)[-top_n:][::-1]
    selected_products = unique_prod[top_idx]
    selected_counts = counts[top_idx]
    selected_avgs = avg_ratings[top_idx]
    def short_label(pid):
        return pid if len(pid) <= 12 else pid[:6] + '…' + pid[-4:]
    labels = [short_label(p) for p in selected_products]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, selected_avgs, color='steelblue', edgecolor='black')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('ProductId', fontsize=14)
    ax.set_ylabel('Average Rating', fontsize=14)
    ax.set_ylim(0, 5.2)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.tick_params(axis='x', rotation=45)
    plt.setp(ax.get_xticklabels(), ha='right')
    for bar, avg, cnt in zip(bars, selected_avgs, selected_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f'{avg:.2f}\n({cnt} ratings)', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_statistics_summary(data, figsize=(14, 10)):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    users = data['UserId']
    products = data['ProductId']
    ratings = data['Rating']
    unique_users, user_counts = np.unique(users, return_counts=True)
    unique_products, product_counts = np.unique(products, return_counts=True)
    ax1 = fig.add_subplot(gs[0, :])
    bins = np.arange(0.5, 6, 0.5)
    counts, edges = np.histogram(ratings, bins=bins)
    ax1.bar(edges[:-1] + 0.25, counts, width=0.4, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Count')
    ax1.set_title('Rating Distribution', fontweight='bold')
    ax1.set_xticks(np.arange(1, 6))
    ax1.grid(axis='y', alpha=0.3)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(user_counts, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel('Ratings per User')
    ax2.set_ylabel('Count')
    ax2.set_title('User Activity', fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(product_counts, bins=30, alpha=0.7, color='mediumseagreen', edgecolor='black')
    ax3.set_xlabel('Ratings per Product')
    ax3.set_ylabel('Count')
    ax3.set_title('Product Popularity', fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(axis='y', alpha=0.3)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    stats_text = f"""
    DATASET STATISTICS

    Total Ratings: {len(data):,}
    Unique Users: {len(unique_users):,}
    Unique Products: {len(unique_products):,}

    Rating Stats:
      Mean: {np.mean(ratings):.3f}
      Median: {np.median(ratings):.1f}
      Std: {np.std(ratings):.3f}

    Sparsity: {100 * (1 - len(data) / (len(unique_users) * len(unique_products))):.4f}%

    User Activity:
      Mean: {np.mean(user_counts):.2f}
      Median: {np.median(user_counts):.1f}

    Product Popularity:
      Mean: {np.mean(product_counts):.2f}
      Median: {np.median(product_counts):.1f}
    """
    ax4.text(0.1, 0.5, stats_text.strip(), fontsize=10, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax5 = fig.add_subplot(gs[2, :2])
    top_n = 20
    sorted_idx = np.argsort(product_counts)[::-1][:top_n]
    ax5.barh(np.arange(top_n), product_counts[sorted_idx], alpha=0.7, color='mediumseagreen', edgecolor='black')
    ax5.set_xlabel('Number of Ratings')
    ax5.set_ylabel('Product Rank')
    ax5.set_title(f'Top {top_n} Most Popular Products', fontweight='bold')
    ax5.invert_yaxis()
    ax5.grid(axis='x', alpha=0.3)
    ax6 = fig.add_subplot(gs[2, 2])
    sorted_counts = np.sort(product_counts)[::-1]
    cumsum_pct = np.cumsum(sorted_counts) / np.sum(sorted_counts) * 100
    ax6.plot(cumsum_pct, color='darkviolet', linewidth=2)
    ax6.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5, label='80%')
    ax6.set_xlabel('Product Rank')
    ax6.set_ylabel('Cumulative %')
    ax6.set_title('Long-tail Effect', fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    plt.suptitle('Data Exploration Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.show()

def plot_kcore_impact_comparison(raw_data_size, processed_data_size, raw_users, processed_users, raw_products, processed_products, raw_sparsity, processed_sparsity, figsize=(14, 10)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Impact of K-core Filtering: Raw vs Processed', fontsize=16, fontweight='bold')
    categories = ['Raw Data', 'Processed Data']
    colors = ['#3498db', '#e74c3c']
    ax1 = axes[0, 0]
    counts = [raw_data_size, processed_data_size]
    bars = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Ratings', fontsize=11)
    ax1.set_title('Total Ratings: Raw vs Processed', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    reduction_pct = (raw_data_size - processed_data_size) / raw_data_size * 100
    ax1.text(0.5, 0.95, f'Reduction: {reduction_pct:.1f}%', transform=ax1.transAxes, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5), fontsize=11, fontweight='bold')
    ax2 = axes[0, 1]
    user_counts = [raw_users, processed_users]
    bars = ax2.bar(categories, user_counts, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Number of Users', fontsize=11)
    ax2.set_title('Total Users: Raw vs Processed', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    user_reduction = (raw_users - processed_users) / raw_users * 100
    ax2.text(0.5, 0.95, f'Reduction: {user_reduction:.1f}%', transform=ax2.transAxes, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5), fontsize=11, fontweight='bold')
    ax3 = axes[1, 0]
    product_counts = [raw_products, processed_products]
    bars = ax3.bar(categories, product_counts, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Number of Products', fontsize=11)
    ax3.set_title('Total Products: Raw vs Processed', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    product_reduction = (raw_products - processed_products) / raw_products * 100
    ax3.text(0.5, 0.95, f'Reduction: {product_reduction:.1f}%', transform=ax3.transAxes, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5), fontsize=11, fontweight='bold')
    ax4 = axes[1, 1]
    sparsity_values = [raw_sparsity * 100, processed_sparsity * 100]
    bars = ax4.bar(categories, sparsity_values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Sparsity (%)', fontsize=11)
    ax4.set_title('Data Sparsity: Raw vs Processed', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.6f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    sparsity_change = processed_sparsity - raw_sparsity
    change_symbol = '↑' if sparsity_change > 0 else '↓'
    ax4.text(0.5, 0.95, f'Change: {change_symbol} {abs(sparsity_change) * 100:.6f}%', transform=ax4.transAxes, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5), fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_rating_distribution_comparison(raw_ratings, processed_ratings, figsize=(14, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Rating Distribution Comparison', fontsize=16, fontweight='bold')
    ax1 = axes[0]
    raw_values, raw_counts = np.unique(raw_ratings, return_counts=True)
    ax1.bar(raw_values, raw_counts, color='#3498db', alpha=0.7, edgecolor='black', width=0.6)
    ax1.set_xlabel('Rating', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Raw Data Rating Distribution', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for v, c in zip(raw_values, raw_counts):
        ax1.text(v, c, f'{c:,}\n({c / len(raw_ratings) * 100:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax2 = axes[1]
    proc_values, proc_counts = np.unique(processed_ratings, return_counts=True)
    ax2.bar(proc_values, proc_counts, color='#e74c3c', alpha=0.7, edgecolor='black', width=0.6)
    ax2.set_xlabel('Rating', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Processed Data Rating Distribution', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for v, c in zip(proc_values, proc_counts):
        ax2.text(v, c, f'{c:,}\n({c / len(processed_ratings) * 100:.1f}%)', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()
    print('Rating Distribution Statistics:')
    print('=' * 60)
    print(f'Raw Data Average Rating: {np.mean(raw_ratings):.4f}')
    print(f'Processed Data Average Rating: {np.mean(processed_ratings):.4f}')
    print(f'Change: {np.mean(processed_ratings) - np.mean(raw_ratings):+.4f}')
    print('=' * 60)

def plot_activity_comparison(raw_user_freq, processed_user_freq, raw_product_freq, processed_product_freq, figsize=(14, 10)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('User & Product Activity: Raw vs Processed', fontsize=16, fontweight='bold')
    ax1 = axes[0, 0]
    ax1.hist(raw_user_freq, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Number of Ratings per User', fontsize=10)
    ax1.set_ylabel('Number of Users (log scale)', fontsize=10)
    ax1.set_yscale('log')
    ax1.set_title('Raw Data: User Activity', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axvline(np.mean(raw_user_freq), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(raw_user_freq):.2f}")
    ax1.legend()
    ax2 = axes[0, 1]
    ax2.hist(processed_user_freq, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Ratings per User', fontsize=10)
    ax2.set_ylabel('Number of Users (log scale)', fontsize=10)
    ax2.set_yscale('log')
    ax2.set_title('Processed Data: User Activity', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axvline(np.mean(processed_user_freq), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(processed_user_freq):.2f}")
    ax2.legend()
    ax3 = axes[1, 0]
    ax3.hist(raw_product_freq, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Number of Ratings per Product', fontsize=10)
    ax3.set_ylabel('Number of Products (log scale)', fontsize=10)
    ax3.set_yscale('log')
    ax3.set_title('Raw Data: Product Popularity', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.axvline(np.mean(raw_product_freq), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(raw_product_freq):.2f}")
    ax3.legend()
    ax4 = axes[1, 1]
    ax4.hist(processed_product_freq, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Number of Ratings per Product', fontsize=10)
    ax4.set_ylabel('Number of Products (log scale)', fontsize=10)
    ax4.set_yscale('log')
    ax4.set_title('Processed Data: Product Popularity', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.axvline(np.mean(processed_product_freq), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(processed_product_freq):.2f}")
    ax4.legend()
    plt.tight_layout()
    plt.show()
    print('Activity Statistics Comparison:')
    print('=' * 80)
    print(f"{'Metric':<35} {'Raw Data':>20} {'Processed Data':>20}")
    print('-' * 80)
    print(f"{'Avg ratings per user':<35} {np.mean(raw_user_freq):>20.2f} {np.mean(processed_user_freq):>20.2f}")
    print(f"{'Min ratings per user':<35} {np.min(raw_user_freq):>20} {np.min(processed_user_freq):>20}")
    print(f"{'Max ratings per user':<35} {np.max(raw_user_freq):>20} {np.max(processed_user_freq):>20}")
    print(f"{'Avg ratings per product':<35} {np.mean(raw_product_freq):>20.2f} {np.mean(processed_product_freq):>20.2f}")
    print(f"{'Min ratings per product':<35} {np.min(raw_product_freq):>20} {np.min(processed_product_freq):>20}")
    print(f"{'Max ratings per product':<35} {np.max(raw_product_freq):>20} {np.max(processed_product_freq):>20}")
    print('=' * 80)

def _lorenz_points(counts):
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    sorted_counts = np.sort(counts)
    cumulative = np.cumsum(sorted_counts)
    cumulative = np.insert(cumulative, 0, 0.0)
    cumulative /= cumulative[-1]
    share = np.linspace(0.0, 1.0, cumulative.size)
    return share, cumulative

def plot_interaction_lorenz(raw_user_counts, processed_user_counts, raw_product_counts, processed_product_counts, figsize=(14, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_users, ax_items = axes
    for label, counts, color in [('Raw', raw_user_counts, '#1F77B4'), ('Processed', processed_user_counts, '#FF7F0E')]:
        share, lorenz = _lorenz_points(counts)
        ax_users.plot(share, lorenz, label=label, color=color, linewidth=2)
    ax_users.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Perfect equality')
    ax_users.set_title('Lorenz Curve – User Activity', fontsize=13, fontweight='bold')
    ax_users.set_xlabel('Share of users', fontsize=11)
    ax_users.set_ylabel('Share of interactions', fontsize=11)
    ax_users.grid(alpha=0.3)
    ax_users.legend()
    for label, counts, color in [('Raw', raw_product_counts, '#1F77B4'), ('Processed', processed_product_counts, '#FF7F0E')]:
        share, lorenz = _lorenz_points(counts)
        ax_items.plot(share, lorenz, label=label, color=color, linewidth=2)
    ax_items.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Perfect equality')
    ax_items.set_title('Lorenz Curve – Item Popularity', fontsize=13, fontweight='bold')
    ax_items.set_xlabel('Share of items', fontsize=11)
    ax_items.set_ylabel('Share of interactions', fontsize=11)
    ax_items.grid(alpha=0.3)
    ax_items.legend()
    plt.tight_layout()
    plt.show()

def plot_kcore_tradeoff_curve(k_values, rating_counts, user_counts, item_counts, baseline_k=None, figsize=(13, 5)):
    k_values = np.asarray(k_values, dtype=np.int64)
    rating_counts = np.asarray(rating_counts, dtype=np.int64)
    user_counts = np.asarray(user_counts, dtype=np.int64)
    item_counts = np.asarray(item_counts, dtype=np.int64)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_left, ax_right = axes
    ax_left.plot(k_values, rating_counts / 1e6, marker='o', color='#1F77B4', linewidth=2)
    ax_left.set_xlabel('k threshold', fontsize=11)
    ax_left.set_ylabel('Ratings (millions)', fontsize=11)
    ax_left.set_title('Rating volume vs. k-core', fontsize=13, fontweight='bold')
    ax_left.grid(alpha=0.3)
    if baseline_k is not None:
        ax_left.axvline(baseline_k, color='gray', linestyle='--', linewidth=1)
        ax_right.axvline(baseline_k, color='gray', linestyle='--', linewidth=1, label=f'Chosen k={baseline_k}')
    ax_right.plot(k_values, user_counts / user_counts[0], marker='s', linewidth=2, color='#2CA02C', label='User coverage')
    ax_right.plot(k_values, item_counts / item_counts[0], marker='^', linewidth=2, color='#D62728', label='Item coverage')
    ax_right.set_xlabel('k threshold', fontsize=11)
    ax_right.set_ylabel('Coverage vs k=1', fontsize=11)
    ax_right.set_title('Coverage drop when tightening k', fontsize=13, fontweight='bold')
    ax_right.grid(alpha=0.3)
    ax_right.legend()
    plt.tight_layout()
    plt.show()

def plot_timestamp_integrity(timestamps, figsize=(14, 5)):
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if timestamps.size == 0:
        return
    timestamps_dt = timestamps.astype('datetime64[s]')
    months = timestamps_dt.astype('datetime64[M]')
    unique_months, month_counts = np.unique(months, return_counts=True)
    sorted_ts = np.sort(timestamps)
    gap_days = np.diff(sorted_ts) / 86400.0
    gap_days = gap_days[gap_days >= 0]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax1, ax2 = axes
    ax1.bar(unique_months.astype('datetime64[D]'), month_counts, color='#1F77B4', width=20)
    ax1.set_title('Monthly rating volume', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Month', fontsize=11)
    ax1.set_ylabel('Ratings per month', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    bins = np.logspace(np.log10(max(1e-3, gap_days.min() + 1e-6)), np.log10(gap_days.max() + 1), 40)
    ax2.hist(gap_days, bins=bins, color='#FF7F0E', alpha=0.8, edgecolor='black')
    ax2.set_xscale('log')
    ax2.set_title('Distribution of inter-event gaps', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Gap length (days, log scale)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()

def plot_bias_violin(user_bias, item_bias, figsize=(10, 5)):
    user_bias = np.asarray(user_bias, dtype=np.float64)
    item_bias = np.asarray(item_bias, dtype=np.float64)
    data = [user_bias, item_bias]
    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    colors = ['#4C72B0', '#DD8452']
    for body, color in zip(parts['bodies'], colors):
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['User bias', 'Item bias'])
    ax.set_ylabel('Bias (mean rating - global mean)', fontsize=11)
    ax.set_title('Bias distribution after debiasing pipeline', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_train_test_distribution(train_user_counts, test_user_counts, train_item_counts, test_item_counts, user_bins=None, item_bins=None, figsize=(14, 5)):
    train_user_counts = np.asarray(train_user_counts, dtype=np.int64)
    test_user_counts = np.asarray(test_user_counts, dtype=np.int64)
    train_item_counts = np.asarray(train_item_counts, dtype=np.int64)
    test_item_counts = np.asarray(test_item_counts, dtype=np.int64)
    if user_bins is None:
        user_bins = np.array([0, 5, 10, 20, 40, 80, 160, 320, 640, 1280, np.inf])
    if item_bins is None:
        item_bins = np.array([0, 5, 10, 25, 50, 100, 200, 400, 800, np.inf])
    def hist_share(counts, bin_edges):
        hist, _ = np.histogram(counts, bins=bin_edges)
        total = hist.sum()
        return hist / total * 100 if total > 0 else hist
    def labels(bin_edges):
        out = []
        for i in range(len(bin_edges) - 1):
            low = int(bin_edges[i])
            high = bin_edges[i + 1]
            out.append(f'>={low}' if np.isinf(high) else f'{low}-{int(high) - 1}')
        return out
    user_labels = labels(user_bins)
    item_labels = labels(item_bins)
    train_user_share = hist_share(train_user_counts, user_bins)
    test_user_share = hist_share(test_user_counts, user_bins)
    train_item_share = hist_share(train_item_counts, item_bins)
    test_item_share = hist_share(test_item_counts, item_bins)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    width = 0.35
    ax_users, ax_items = axes
    x_users = np.arange(len(user_labels))
    ax_users.bar(x_users - width / 2, train_user_share, width, label='Train', color='#1F77B4')
    ax_users.bar(x_users + width / 2, test_user_share, width, label='Test', color='#FF7F0E')
    ax_users.set_xticks(x_users)
    ax_users.set_xticklabels(user_labels, rotation=30, ha='right')
    ax_users.set_ylabel('Share of users (%)', fontsize=11)
    ax_users.set_title('User activity buckets', fontsize=13, fontweight='bold')
    ax_users.legend()
    ax_users.grid(axis='y', alpha=0.3)
    x_items = np.arange(len(item_labels))
    ax_items.bar(x_items - width / 2, train_item_share, width, label='Train', color='#1F77B4')
    ax_items.bar(x_items + width / 2, test_item_share, width, label='Test', color='#FF7F0E')
    ax_items.set_xticks(x_items)
    ax_items.set_xticklabels(item_labels, rotation=30, ha='right')
    ax_items.set_ylabel('Share of items (%)', fontsize=11)
    ax_items.set_title('Item popularity buckets', fontsize=13, fontweight='bold')
    ax_items.legend()
    ax_items.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
