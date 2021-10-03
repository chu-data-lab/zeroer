from collections import defaultdict

from pandas import merge
import py_entitymatching as em


""" This python file contains blocking functions
specific to table pairs. For example block_fodors_zagat
is the blocking function for Tables fodors and zagat.
Functionality: creates initial set of tuple pairs for
two tables.
"""


def verify_blocking_ground_truth(A, B, block_df, duplicates_df, objectify=False):
    num_duplicates_missed = 0
    duplicates_df.columns = ["ltable_id", "rtable_id"]
    # Sometimes pandas / Magellan puts some columns as objects instead of numeric/string. In this case, we will force this to join appropriately
    if objectify:
        duplicates_df = duplicates_df.astype(object)

    # Intuition: merge function joints two data frames. The outer option creates a number of NaN rows when
    # some duplicates are missing in the blocked_df
    # we leverage the fact that len gives all rows while count gives non-NaN to compute the missing options
    merged_df = block_df.merge(duplicates_df, left_on=["ltable_id", "rtable_id"], right_on=["ltable_id", "rtable_id"],
                               how='outer')
    num_duplicates_missed = len(merged_df) - merged_df["_id"].count()
    total_duplicates = len(duplicates_df)

    print("Ratio saved=", 1.0 - float(len(block_df)) / float(len(A) * len(B)))
    print("Totally missed:", num_duplicates_missed, " out of ", total_duplicates)


def blocking_for_citeseer_dblp(A,B):
    #A = em.read_csv_metadata("citeseer_sample.csv", key="id", encoding='utf-8')
    #B = em.read_csv_metadata("dblp_sample.csv", key="id", encoding='utf-8')
    attributes = ['id', 'title', 'authors', 'journal', 'month', 'year', 'publication_type']

    ob = em.OverlapBlocker()
    C1 = ob.block_tables(A, B, 'title', 'title', word_level=True, overlap_size=2, show_progress=True,
                         l_output_attrs=attributes, r_output_attrs=attributes)
    return C1
    #verify_blocking_ground_truth(A, B, C1, matches_df_head)

#fodors.csv and zagats.csv
def block_fodors_zagats(A, B):
    ob = em.OverlapBlocker()
    C = ob.block_tables(A, B, 'name', 'name', l_output_attrs=['name', 'addr', 'city', 'phone'],  r_output_attrs=['name', 'addr', 'city', 'phone'],
        overlap_size=1, show_progress=False)
    return C


#babies_r_us.csv and buy_buy_baby.csv
def block_baby_products(A, B):
    ob = em.OverlapBlocker()
    # attributes = ['title', 'price', 'category', 'company_struct', 'brand', 'weight', 'length', 'width', 'height', 'fabrics', 'colors', 'materials']
    attributes = ['title', 'price', 'is_discounted', 'category', 'company_struct']
    # C = ob.block_tables(A, B, 'title', 'title', l_output_attrs=attributes,  r_output_attrs=attributes,
    #     overlap_size=3, show_progress=False)
    C = ob.block_tables(A, B, 'title', 'title', word_level = True, overlap_size = 4, show_progress = True, l_output_attrs = attributes, r_output_attrs = attributes)
    return C


#barnes_and_noble.csv and half.csv
def block_books(A, B):
    #assumes some preprocessing is done:
    #Specifically in half.csv : NewPrice  => Price

    ob = em.OverlapBlocker()
    # attributes = ['Title', 'Price', 'Author', 'ISBN13', 'Publisher', 'Publication_Date', 'Pages', 'Dimensions']
    attributes = ['Title', 'Author', 'ISBN13', 'Publisher', 'Publication_Date', 'Pages', 'Dimensions']
    # C = ob.block_tables(A, B, 'Title', 'Title', l_output_attrs=attributes,  r_output_attrs=attributes,
    #     overlap_size=1, show_progress=False)
    C = ob.block_tables(A, B, 'Title', 'Title', word_level=True, overlap_size=4, show_progress=True,
                        l_output_attrs=attributes, r_output_attrs=attributes)
    return C


#yellow_pages.csv and yelp.csv
def block_restaurants(A, B):
    #assumes some preprocessing is done:
    #Specifically in half.csv : NewPrice  => Price

    ob = em.OverlapBlocker()
    attributes = ['name', 'address', 'city', 'state', 'zipcode', 'phone']
    # C = ob.block_tables(A, B, 'name', 'name', l_output_attrs=attributes,  r_output_attrs=attributes,
    #     overlap_size=1, show_progress=False)
    C = ob.block_tables(A, B, 'name', 'name', word_level=True, overlap_size=4, show_progress=True,
                        l_output_attrs=attributes, r_output_attrs=attributes)
    return C


#dblp.csv and ACM.csv
def block_dblp_acm(A, B):
    ab = em.AttrEquivalenceBlocker()
    C = ab.block_tables(A, B, l_block_attr='year', r_block_attr='year', l_output_attrs=["title","authors","venue","year"],
        r_output_attrs=["title","authors","venue","year"], allow_missing=False)
    ob = em.OverlapBlocker()
    #=================>results in a candidate set of size 46K with 5 missing duplicates out of 2224
    C2 = ob.block_candset(C, 'title', 'title', word_level=True, overlap_size=2, show_progress=True)
    return C2


#dblp.csv and google_scholar.csv
def block_dblp_scholar(A, B):
    ob = em.OverlapBlocker()
    attributes = ["id","title","authors","venue","year"]
    #C1 = ob.block_tables(A, B, 'title', 'title', word_level=True, overlap_size=3, show_progress=True, l_output_attrs=attributes, r_output_attrs=attributes)
    #=================>results in a candidate set of size 1.2M with 178 missing duplicates out of 5347
    C2 = ob.block_tables(A, B, 'title', 'title', word_level=True, overlap_size=4, show_progress=True, l_output_attrs=attributes, r_output_attrs=attributes)
    #=================>results in a candidate set of size 135K with 467 missing duplicates out of 5347
    return C2

def block_rotten_imdb(A, B):
    ob = em.OverlapBlocker()
    attributes = set(A.columns)
    attributes.remove("id")
    attributes = list(attributes.intersection(set(B.columns)))
    #C1 = ob.block_tables(A, B, 'title', 'title', word_level=True, overlap_size=3, show_progress=True, l_output_attrs=attributes, r_output_attrs=attributes)
    #=================>results in a candidate set of size 1.2M with 178 missing duplicates out of 5347
    C2 = ob.block_tables(A, B, 'Name', 'Name', word_level=True, overlap_size=2, show_progress=True, l_output_attrs=attributes, r_output_attrs=attributes)
    #=================>results in a candidate set of size 135K with 467 missing duplicates out of 5347
    return C2


#abt.csv and buy.csv
def block_abt_buy(A, B):
    try:
        B["description"] = B["description"] + " " + B["manufacturer"]
    except:
        print()
    ob = em.OverlapBlocker()
    #=================>results in a candidate set of size 164K with 6 missing duplicates out of 1097
    C = ob.block_tables(A, B, "name", "name", word_level=True, overlap_size=1,
    l_output_attrs=["name","description","price"], r_output_attrs=["name","description","price"], show_progress=True, allow_missing=False)
    return C


#walmart.csv and amazon.csv
def block_walmart_amazon_(A, B):
    #assumes some preprocessing is done:
    #Specifically in amazon.csv : a.    pcategory2  => groupname , b.    { proddescrshort,proddescrlong } => shortdescr,longdescr

    ob = em.OverlapBlocker()

    #C1 = ob.block_tables(ltable, rtable, 'title', 'title', word_level=True, overlap_size=2)
    #=================>results in a candidate set of size 1.1M with 20 missing duplicates out of 1154
    #blocking_utils.verify_blocking_ground_truth(dataset_name, C1)

    attributes = ['brand', 'groupname', 'title', 'price', 'shortdescr', 'longdescr', 'imageurl', 'modelno', 'shipweight', 'dimensions']
    C2 = ob.block_tables(A, B, 'title', 'title', word_level=True, overlap_size=3, l_output_attrs=attributes, r_output_attrs=attributes, show_progress=True, allow_missing=True)
    #=================>results in a candidate set of size 278K with 84 missing duplicates out of 1154
    #blocking_utils.verify_blocking_ground_truth(dataset_name, C2)

    return C2

#walmart.csv and amazon.csv
def block_walmart_amazon(A, B):
    #assumes some preprocessing is done:
    #Specifically in amazon.csv : a.    pcategory2  => groupname , b.    { proddescrshort,proddescrlong } => shortdescr,longdescr

    ob = em.OverlapBlocker()

    #C1 = ob.block_tables(ltable, rtable, 'title', 'title', word_level=True, overlap_size=2)
    #=================>results in a candidate set of size 1.1M with 20 missing duplicates out of 1154
    #blocking_utils.verify_blocking_ground_truth(dataset_name, C1)

    r_attributes = ["title","proddescrshort","brand","price","dimensions","shipweight"]
    l_attributes = ["title","shortdescr","brand","price","dimensions","shipweight"]

    if not set(r_attributes).issubset(B.columns): # fix in case A B are the same dataset
        r_attributes = l_attributes
    if not set(l_attributes).issubset(A.columns):
        l_attributes = r_attributes
    #attributes = ['brand', 'groupname', 'title', 'price', 'shortdescr', 'longdescr', 'imageurl', 'modelno', 'shipweight', 'dimensions']
    C2 = ob.block_tables(A, B, 'title', 'title', word_level=True, overlap_size=2, l_output_attrs=l_attributes, r_output_attrs=r_attributes, show_progress=True, allow_missing=True)
    #=================>results in a candidate set of size 278K with 84 missing duplicates out of 1154
    #blocking_utils.verify_blocking_ground_truth(dataset_name, C2)
    return C2

def block_wa(A, B):
    #assumes some preprocessing is done:
    #Specifically in amazon.csv : a.    pcategory2  => groupname , b.    { proddescrshort,proddescrlong } => shortdescr,longdescr

    ob = em.OverlapBlocker()

    #C1 = ob.block_tables(ltable, rtable, 'title', 'title', word_level=True, overlap_size=2)
    #=================>results in a candidate set of size 1.1M with 20 missing duplicates out of 1154
    #blocking_utils.verify_blocking_ground_truth(dataset_name, C1)

    r_attributes = ["title","category","brand","modelno","price"]
    l_attributes = ["title","category","brand","modelno","price"]

    if not set(r_attributes).issubset(B.columns): # fix in case A B are the same dataset
        r_attributes = l_attributes
    if not set(l_attributes).issubset(A.columns):
        l_attributes = r_attributes
    #attributes = ['brand', 'groupname', 'title', 'price', 'shortdescr', 'longdescr', 'imageurl', 'modelno', 'shipweight', 'dimensions']
    C2 = ob.block_tables(A, B, 'title', 'title', word_level=True, overlap_size=2, l_output_attrs=l_attributes, r_output_attrs=r_attributes, show_progress=True, allow_missing=True)
    #=================>results in a candidate set of size 278K with 84 missing duplicates out of 1154
    #blocking_utils.verify_blocking_ground_truth(dataset_name, C2)
    return C2

#amazon.csv and GoogleProducts.csv
def block_amazon_googleproducts(A, B):
    ob = em.OverlapBlocker()
    #=================>results in a candidate set of size 400K with 6 missing duplicates out of 1300
    C = ob.block_tables(A, B, "title", "title", word_level=True, overlap_size=1, l_output_attrs=["title","description","manufacturer","price"], r_output_attrs=["title","description","manufacturer","price"], show_progress=True, allow_missing=False)
    return C

def block_songs(A, B):
    ob = em.OverlapBlocker()
    #=================>results in a candidate set of size 400K with 6 missing duplicates out of 1300
    C = ob.block_tables(A, B, "title", "title", word_level=True, overlap_size=1,
                        l_output_attrs=["title","release","artist_name","duration","artist_familiarity","artist_hotttnesss","year"],
                        r_output_attrs=["title","release","artist_name","duration","artist_familiarity","artist_hotttnesss","year"],
                        show_progress=True, allow_missing=False,n_jobs=8)
    return C

def generic_blocking_func(A, B):
    A_prefix = A.add_prefix('ltable_')
    B_prefix = B.add_prefix('rtable_')
    A_prefix['key'] = 1
    B_prefix['key'] = 1
    final = merge(A_prefix, B_prefix,on='key', suffixes=('', ''))
    final = final.drop(columns=['key'])
    final = final.reset_index()
    final = final.rename(columns={'index': '_id'})
    print (list(final))
    return final


blocking_functions_mapping = defaultdict(str)
blocking_functions_mapping["fodors_zagats"] = block_fodors_zagats
blocking_functions_mapping["abt_buy"] = block_abt_buy
blocking_functions_mapping["dblp_acm"] = block_dblp_acm
blocking_functions_mapping["dblp_scholar"] = block_dblp_scholar
blocking_functions_mapping["amazon_googleproducts"] = block_amazon_googleproducts
blocking_functions_mapping["walmart_amazon"] = block_walmart_amazon
blocking_functions_mapping["songs"] = block_songs
blocking_functions_mapping["citations"] = blocking_for_citeseer_dblp

blocking_functions_mapping["dblp_citeseer"] = generic_blocking_func
blocking_functions_mapping["imdb_omdb"] = generic_blocking_func
blocking_functions_mapping["rotten_imdb"] = block_rotten_imdb

blocking_functions_mapping["cora"] = generic_blocking_func
blocking_functions_mapping["synthetic"] = generic_blocking_func
blocking_functions_mapping["books"] = block_books
blocking_functions_mapping["baby_products"] = block_baby_products
blocking_functions_mapping["restaurants"] = block_restaurants
blocking_functions_mapping['wa'] = block_wa