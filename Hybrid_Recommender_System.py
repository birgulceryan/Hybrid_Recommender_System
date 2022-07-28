##########################################################
#Hybrid Recommender System
##########################################################

##########################################################
#İş Problemi
###########################################################
#ID'si verilen kullanıcı için item-based ve user-based recommender yöntemlerini kullanarak 10 film önerisi yapınız.
###########################################################

###########################################################
#Veri Seti Hikayesi
###########################################################

#Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır. İçerisinde filmler ile birlikte bu filmlere yapılan derecelendirme puanlarını barındırmaktadır.
#27.278 filmde 2.000.0263 derecelendirme içermektedir. Bu veri seti ise 17 Ekim 2016 tarihinde oluşturulmuştur.
#138.493 kullanıcı ve 09 Ocak 1995 ile 31 Mart 2015 tarihleri arasında verileri içermektedir. Kullanıcılar rastgele seçilmiştir.
#Seçilen tüm kullanıcıların en az 20 filme oy verdiği bilgisi mevcuttur.

##########################################################
#Değişkenler
##########################################################
#movie.csv
#movieId : Eşsiz film numarası
#title : Film adı
#genres : Tür
#rating.csv
#userid : Eşsiz kullanıcı numarası. (UniqueID)
#movieId : Eşsiz film numarası. (UniqueID)
#rating : Kullanıcı tarafından filme verilen puan
#timestamp : Değerlendirme tarihi
#########################################################

#########################################################
#Görev 1: Veri Hazırlama
#########################################################

########
#Adım 1: movie, rating veri setlerini okutunuz.
########

import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
movie = pd.read_csv('Hybrid_Recommender_System/movie.csv')
rating = pd.read_csv('Hybrid_Recommender_System/rating.csv')

#########
#Adım 2: Rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.
#########

df = movie.merge(rating, how="left", on="movieId")
df.head()
df.shape        #20000797

#########
#Adım 3: Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listede tutunuz ve veri setinden çıkartınız.
#########

df["title"].nunique()    #27262

df["title"].value_counts().head()     #her bir film için kaç tane değerlendirme var

comment_counts = pd.DataFrame(df["title"].value_counts())

rare_movies = comment_counts[comment_counts["title"] <= 1000].index

common_movies = df[~df["title"].isin(rare_movies)]
common_movies
common_movies.shape     #17766015

common_movies["title"].nunique()  #3159
df["title"].nunique()             #eski hali 27262

#########
#Adım 4: index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe için pivot table oluşturunuz.
#########

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df
user_movie_df.shape  #satır 138493 user
user_movie_df.columns  # sütunlarda 3159 film

#########
#Adım 5: Yapılan tüm işlemleri fonksiyonlaştırınız.
#########

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('Hybrid_Recommender_System/movie.csv')
    rating = pd.read_csv('Hybrid_Recommender_System/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

#########################################################################
#Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#########################################################################
##########
#Adım 1: Rastgele bir kullanıcı id’si seçiniz.
##########

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
random_user     #28941

#########
#Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
#########

random_user
user_movie_df
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df

#########
#Adım 3: Seçilen kullanıcıların oy kullandığı filmleri movies_watched adında bir listeye atayınız.
#########

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
movies_watched

user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Forrest Gump (1994)"]

len(movies_watched)

########################################################################################
#Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
########################################################################################
########
#Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturunuz.
########

movies_watched_df = user_movie_df[movies_watched]    #sadece izlenen filmler bilgisi

#########
#Adım 2: Her bir kullancının seçili user'in izlediği filmlerin kaçını izlediğini bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
#########

user_movie_count = movies_watched_df.T.notnull().sum()   #her bir kullanıcının kaç tane film izlediği
user_movie_count

user_movie_count = user_movie_count.reset_index()
user_movie_count
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count

#########
#Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı id’lerinden users_same_movies adında bir liste oluşturunuz.
##########

user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]
users_same_movies

#yüzdeliği belirleyerek yapma işlemi:
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
# perc = len(movies_watched) * 60 / 100

perc = len(movies_watched) * 60 / 100
perc
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
users_same_movies

###############################################################################
#Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
###############################################################################
#########
#Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.
#########

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

#########
#Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
#########

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ['user_id_1', 'user_id_2']

corr_df = corr_df.reset_index()

#########
#Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
#########

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][

    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

#########
#Adım 4: top_users dataframe’ine rating veri seti ile merge ediniz.
#########

rating = pd.read_csv('Hybrid_Recommender_System/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

#############################################################################################
#Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
############################################################################################3
########
#Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.
########

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

#########
#Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir dataframe oluşturunuz.
#########

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})


recommendation_df = recommendation_df.reset_index()

# veya recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"}).reset_index()

#########
#Adım 3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
#########

recommendation_df[recommendation_df["weighted_rating"] > 3.5]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

########
#Adım 4: movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz.
########

movie = pd.read_csv('Hybrid_Recommender_System/movie.csv')      # sinanın beğenebileceği filmler
movies_to_be_recommend.merge(movie[["movieId", "title"]]).sort_values("weighted_rating", ascending=False)

#Item Based Recommendation
##########
#Görev 1: Kullanıcının izlediği en son ve en yüksek puan verdiği filme göre item-based öneri yapınız.
##########
#Adım 1: movie, rating veri setlerini okutunuz.
##########
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('Hybrid_Recommender_System/movie.csv')
    rating = pd.read_csv('Hybrid_Recommender_System/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

#########
#Adım 2: Seçili kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
#########

user_id = 28491
movie_id = rating[(rating["userId"] == user_id) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]
movie_id
movies_to_be_recommend.merge(movie[["movieId", "title"]])[:5]['title'].to_list()

########
#Adım 3: User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
########
movie_name = movie[movie['movieId'] == movie_id]['title'].values[0]
movie_name = user_movie_df[movie_name]

#Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
moveis_from_item_based = user_movie_df.corrwith(movie_name).sort_values(ascending=False)

#Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.

moveis_from_item_based[1:6].index.to_list()
