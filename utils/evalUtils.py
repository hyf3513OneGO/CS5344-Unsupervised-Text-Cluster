from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score

def evaluate_ari(true_labels, pred_labels):
    '''
    adjust rand index,closer to 1 ->better
    '''
    ari = adjusted_rand_score(true_labels, pred_labels)
    print(f'Adjusted Rand Index (ARI): {ari:.4f}')
    return ari


def evaluate_nmi(true_labels, pred_labels):
    '''
    normalized mutual info,bigger->better
    '''
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    print(f'Normalized Mutual Information (NMI): {nmi:.4f}')
    return nmi

def evaluate_silhouette(X, labels):
    '''
    silhouette,bigger->better
    '''
    silhouette = silhouette_score(X, labels)
    print(f'Silhouette Score: {silhouette:.4f}')
    return silhouette
