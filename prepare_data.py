import json
import os.path
import subprocess

import polars as pl

pl.Config.set_fmt_str_lengths(50)

DATASET_DIR = "TMDB-5000"


def download_dataset():
    if not os.path.exists(DATASET_DIR):
        subprocess.run(["curl", "-O", "https://storage.ml-workout.pl/datasets/TMDB-5000.tar.bz2"])
        subprocess.run(["tar", "-xjf", "TMDB-5000.tar.bz2"])


def preprocess_dataset(credits_df, movies_df):
    def movie_cast_as_text(cast_json, top_n=7):
        cast = json.loads(cast_json)
        formatted = ["- {role} played by {actor}. ".format(role=c["character"], actor=c["name"]) for c in cast[:top_n]]
        return "Cast:\n" + "\n".join(formatted)

    def movie_genres_as_text(genres_json, top_n=3):
        genres = json.loads(genres_json)
        formatted = "Movie genres:\n" + "\n".join(f"- {g['name'].lower()}" for g in genres[:top_n])
        return formatted

    # join credits and movies
    tmdb = movies_df.select("title", "overview", "release_date", "genres", "id").join(
        credits_df.select("cast", "movie_id"), left_on="id", right_on="movie_id"
    )
    # preprocess
    # output text dataframe consists of columns: [id, text]
    tmdb_texts = tmdb.select(
        "id",
        pl.struct(["title", "overview", "release_date", "genres", "cast"])
        .apply(
            lambda row: f"""passage: Title: {row['title']}
    Summary: {row['overview']}
    Released on: {row['release_date']}
    {movie_cast_as_text(row['cast'])}
    {movie_genres_as_text(row['genres'])}
    """.strip()
        )
        .alias("text"),
    )
    return tmdb_texts


def save_as_json_files(tmdb_texts_df, output_dir="./data"):
    def save_row_as_json(row):
        row_id, row_text = row
        row_dict = {"id": row_id, "text": row_text}

        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{row_id}.json", "w") as f:
            json.dump(row_dict, f)
        return 1

    result = tmdb_texts_df.apply(save_row_as_json)
    assert result["map"].sum() == tmdb_texts_df.shape[0], "Some rows are not exported properly!"


def main():
    download_dataset()

    # read into polars dataframe
    credits_df = pl.read_csv(f"./{DATASET_DIR}/tmdb_5000_credits.csv")
    movies_df = pl.read_csv(f"./{DATASET_DIR}/tmdb_5000_movies.csv", infer_schema_length=10000)

    tmdb_texts_df = preprocess_dataset(credits_df, movies_df)
    save_as_json_files(tmdb_texts_df)


if __name__ == "__main__":
    main()
