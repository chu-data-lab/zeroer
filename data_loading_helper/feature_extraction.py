import pandas as pd
import numpy as np
import py_entitymatching as em
from .magellan_modified_feature_generation import get_features


#Given a CANDIDATE SET and the list of ACTUAL duplicates (duplicates_df),
#this function adds the 1/0 labels (column name = GOLD) to the candset dataframe
def add_labels_to_candset(duplicates_df, candset_df, ltable_df, rtable_df):
    #We are overwriting column names - but thats okay as this is not used anywhere else.
    duplicates_df.columns = ["ltable_id", "rtable_id"]

    #We merged two DF based on the common attributes. The indicator 'gold' takes three values both, left_only, right_only
    df_with_gold = pd.merge(candset_df, duplicates_df, on=['ltable_id', 'rtable_id'], how='left', indicator='gold')

    #If it is present in both, then it is a duplicate and we set it to 1 and 0 otherwise
    df_with_gold['gold'] = np.where(df_with_gold.gold == 'both', 1, 0)

    #This is to handle some Magellan issues
    em.set_key(df_with_gold, '_id')
    em.set_property(df_with_gold,'ltable', ltable_df)
    em.set_property(df_with_gold,'rtable', rtable_df)
    em.set_property(df_with_gold,'fk_ltable', "ltable_id")
    em.set_property(df_with_gold,'fk_rtable', "rtable_id")

    return df_with_gold

def get_features_for_type(column_type):
    """
    Get features to be generated for a type
    """
    # First get the look up table
    lookup_table = dict()

    # Features for type str_eq_1w
    lookup_table['STR_EQ_1W'] = [('lev_dist'), ('lev_sim'), ('jaro'),
                                ('jaro_winkler'),
                                 ('exact_match'),
                                 ('jaccard', 'qgm_3', 'qgm_3')]

    # Features for type str_bt_1w_5w
    lookup_table['STR_BT_1W_5W'] = [('jaccard', 'qgm_3', 'qgm_3'),
                                    ('cosine', 'dlm_dc0', 'dlm_dc0'),
                                    ('jaccard', 'dlm_dc0', 'dlm_dc0'),
                                    ('monge_elkan'), ('lev_dist'), ('lev_sim'),
                                    ('needleman_wunsch'),
                                    ('smith_waterman')]  # dlm_dc0 is the concrete space tokenizer

    # Features for type str_bt_5w_10w
    lookup_table['STR_BT_5W_10W'] = [('jaccard', 'qgm_3', 'qgm_3'),
                                     ('cosine', 'dlm_dc0', 'dlm_dc0'),
                                     ('monge_elkan'), ('lev_dist'), ('lev_sim')]

    # Features for type str_gt_10w
    lookup_table['STR_GT_10W'] = [('jaccard', 'qgm_3', 'qgm_3'),
                                  ('cosine', 'dlm_dc0', 'dlm_dc0')]

    # Features for NUMERIC type
    lookup_table['NUM'] = [('exact_match'), ('abs_norm'), ('lev_dist'),
                           ('lev_sim')]

    # Features for BOOLEAN type
    lookup_table['BOOL'] = [('exact_match')]

    # Features for un determined type
    lookup_table['UN_DETERMINED'] = []
    # Based on the column type, return the feature functions that should be
    # generated.
    if column_type is 'str_eq_1w':
        features = lookup_table['STR_EQ_1W']
    elif column_type is 'str_bt_1w_5w':
        features = lookup_table['STR_BT_1W_5W']
    elif column_type is 'str_bt_5w_10w':
        features = lookup_table['STR_BT_5W_10W']
    elif column_type is 'str_gt_10w':
        features = lookup_table['STR_GT_10W']
    elif column_type is 'numeric':
        features = lookup_table['NUM']
    elif column_type is 'boolean':
        features = lookup_table['BOOL']
    elif column_type is 'un_determined':
        features = lookup_table['UN_DETERMINED']
    else:
        raise TypeError('Unknown type')
    return features


def extract_features(ltable_df, rtable_df, candset_df):
    tokenizers = em.get_tokenizers_for_matching()
    sim_functions = em.get_sim_funs_for_matching()
    left_attr_types = em.get_attr_types(ltable_df)
    right_attr_types = em.get_attr_types(rtable_df)
    correspondences = em.get_attr_corres(ltable_df, rtable_df)

    feature_dict_list = []
    attribute_type_rank = {'boolean':1, 'numeric':2, 'str_eq_1w':3, 'str_bt_1w_5w':4, 'str_bt_5w_10w':5, 'str_gt_10w':6, 'un_determined':7}
    for c in correspondences['corres']:
        if left_attr_types[c[0]] != right_attr_types[c[1]]:
            if attribute_type_rank[left_attr_types[c[0]]] < attribute_type_rank[right_attr_types[c[1]]]:
                left_attr_types[c[0]] = right_attr_types[c[1]]
            else:
                right_attr_types[c[1]] = left_attr_types[c[0]]

    feature_records = get_features(ltable_df,rtable_df,left_attr_types, right_attr_types, correspondences, tokenizers, sim_functions)
    #Remove all features based on id - they are often useless
    feature_records = feature_records[feature_records.left_attribute !='id']
    feature_records.reset_index(inplace=True,drop=True)

    distance_functions = ["lev_dist", "rdf"]
    non_normalized_functions = ["aff", "sw", "swn", "nmw"]
    keep_features = [True]*feature_records.shape[0]
    for i in range(feature_records.shape[0]):
        feature = feature_records.loc[i,"feature_name"]
        for func in distance_functions + non_normalized_functions:
            if func in feature:
                keep_features[i] = False
    feature_records = feature_records.loc[keep_features,:]

    print("\n\nExtracting the full set of features:")
    candset_features_df = em.extract_feature_vecs(candset_df,feature_table=feature_records,attrs_after='gold',show_progress=True,n_jobs=-1)
    candset_features_df.fillna(value=0, inplace=True)

    return candset_features_df



def extract_features_auto(ltable_df, rtable_df, candset_df):
    feature_list = em.get_features_for_matching(ltable_df,rtable_df,validate_inferred_attr_types=False)
    #Remove all features based on id - they are often useless
    feature_list = feature_list[feature_list.left_attribute !='id']

    print("\n\nExtracting the full set of features:")
    candset_features_df = em.extract_feature_vecs(candset_df,feature_table=feature_list,attrs_after='gold',show_progress=True)
    candset_features_df.fillna(value=0, inplace=True)

    return candset_features_df


#High level function which just adds labels and the complete set of features to candset
def gather_features_and_labels(ltable_df, rtable_df, labels_df, candset_df):
    labels_df.columns = ["ltable_id", "rtable_id"]
    labels_df["ltable_id"] = labels_df["ltable_id"].astype(str)
    labels_df["rtable_id"] = labels_df["rtable_id"].astype(str)
    candset_df["ltable_id"] = candset_df["ltable_id"].astype(str)
    candset_df["rtable_id"] = candset_df["rtable_id"].astype(str)
    ltable_df["id"] = ltable_df["id"].astype(str)
    rtable_df["id"] = rtable_df["id"].astype(str)
    candset_df = add_labels_to_candset(labels_df, candset_df, ltable_df, rtable_df)
    candset_features_df = extract_features(ltable_df, rtable_df, candset_df)
        
    return candset_features_df


#Filter out bad features (non similarity, non distance, singular valued)
def gather_similarity_features(candset_features_df, avged = False):
    distance_functions = ["lev_dist", "rdf"]
    non_normalized_functions = ["aff", "sw", "swn", "nmw"]
    
    cols = candset_features_df.columns
    cols_to_be_dropped = []
    for col in cols:
        for func in distance_functions + non_normalized_functions:
            if func in col:
                cols_to_be_dropped.append(col)
                break

    candset_similarity_features_df = candset_features_df.drop(cols_to_be_dropped, axis=1)
    similarity_features_df = candset_similarity_features_df.drop(['gold', '_id', 'ltable_id', 'rtable_id'], axis=1)
    
    # Dropping columns that have only one value
    cols_to_be_dropped = []
    col_count_map = similarity_features_df.nunique()
    for col in similarity_features_df.columns:
        if col_count_map[col] == 1:
            cols_to_be_dropped.append(col)
    similarity_features_df = similarity_features_df.drop(cols_to_be_dropped, axis=1)
    

    if (avged==False):
        return similarity_features_df

    
    headers= similarity_features_df.columns.values

    attributes = []
    for h in headers:
        arr = h.split("_")
        attributes.append(arr[0])
    attributes = set(attributes)

    avged_df = pd.DataFrame()
    
    for attribute in attributes:
        #print("\nFeatures for attribute:", attribute)
        matches = np.zeros(candset_features_df.shape[0])
        counts = 0
        for h in headers:
            if attribute in h:
                #print(h)
                matches = np.add(matches, candset_features_df[h].values)
                counts += 1
        matches = matches/counts
        avged_df[attribute] = matches
    
    return avged_df