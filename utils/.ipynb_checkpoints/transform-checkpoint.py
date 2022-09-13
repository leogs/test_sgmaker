import pandas as pd
import numpy as np

from utils import eda

def load_data(X: list,
              y = None) -> (pd.DataFrame, pd.DataFrame):
    
    X = pd.json_normalize(X)
    if y is not None:
        y = y
        return X, y
    else:
        return X

def get_base_cols() -> list:
    return ['base_price', 'accepts_mercadopago',
            'automatic_relist', 'initial_quantity',
            'sold_quantity', 'shipping.local_pick_up',
            'shipping.free_shipping', 'create_update_diff_days',
            'MLATB', 'MLAWC', 'MLAMO', 'MLAOT',
            'MLAMC', 'MLACD', 'MLADC', 'MLAWT',
            'MLAMP', 'MLABC', 'variations_count',
            'has_attributes', 'dragged_bids_and_visits',
            'good_quality_thumbnail', 'dragged_visits',
            'free_relist', 'poor_quality_thumbnail',
            'pictures_amnt', 'diff_start_stop_time',
            'listing_type_id', 'shipping.mode']
    
def transform(X: list, y = None, load=True) -> pd.DataFrame:
    if load:
        X, y = load_data(X, y)
    
    dt_cols = ['date_created', 'last_updated']
    X[dt_cols] = (
        X[dt_cols].apply(pd.to_datetime, errors='coerce'))
        
    X['create_update_diff_days'] = (
        X['last_updated'] - X['date_created']).dt.days
        
    X['deal_ids'] = X['deal_ids'].apply(
        lambda x: x[0] if len(x) > 0 else np.nan)
        
    X['non_mercado_pago_payment_methods'] = (
        X['non_mercado_pago_payment_methods']
        .apply(lambda x: [d.get('id') for d in x]))
        
    X = eda.one_hot_encode_list(X, 'non_mercado_pago_payment_methods')
        
    X['variations_count'] = X['variations'].apply(lambda x: len(x))
            
    X['has_attributes'] = X['attributes'].apply(
        lambda x: 1 if len(x) > 0 else 0)
        
    X = eda.one_hot_encode_list(X, 'tags')
    
    X['pictures_amnt'] = X['pictures'].apply(lambda x: len(x))
        
    X['diff_start_stop_time'] = X['start_time'] - X['stop_time']
    
    X['listing_type_id'] = X['listing_type_id'].fillna('None')
    X['shipping.mode'] = X['shipping.mode'].fillna('None')
        
    #X = pd.get_dummies(X, columns=['listing_type_id', 'shipping.mode'])
        
    X = X.reindex(columns=get_base_cols()).fillna(0)
    
    X['listing_type_id'] = X['listing_type_id'].astype('category')
    X['shipping.mode'] = X['shipping.mode'].astype('category')
    
    X[['accepts_mercadopago',
       'automatic_relist',
       'shipping.local_pick_up',
       'shipping.free_shipping']] = (X[['accepts_mercadopago',
                                        'automatic_relist',
                                        'shipping.local_pick_up',
                                        'shipping.free_shipping']].astype(int))
    
    if y is not None:
        y = pd.Series(y).map({'new': 1, 'used': 0}).values
        return X, y
    else:
        return X    