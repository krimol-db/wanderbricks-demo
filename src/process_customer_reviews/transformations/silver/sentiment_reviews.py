from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.materialized_view(
    comment="Verified reviews enriched with AI sentiment analysis and a composite sentiment score"
)
def sentiment_reviews():
    verified = spark.read.table("verified_reviews")

    # Use AI to analyze sentiment of the written comment
    with_sentiment = verified.withColumn(
        "ai_sentiment", F.expr("ai_analyze_sentiment(comment)")
    )

    # Map AI sentiment to a numeric weight
    sentiment_weight = (
        F.when(F.col("ai_sentiment") == "positive", 1.0)
        .when(F.col("ai_sentiment") == "neutral", 0.0)
        .when(F.col("ai_sentiment") == "mixed", 0.0)
        .otherwise(-1.0)  # negative
    )

    # Composite score: normalize rating to [-1, 1] and average with sentiment weight
    # Rating is 1.0-5.0, normalized: (rating - 3) / 2 gives range [-1, 1]
    normalized_rating = (F.col("rating") - 3) / 2

    enriched = with_sentiment.withColumn(
        "sentiment_weight", sentiment_weight
    ).withColumn(
        "composite_score", F.round((normalized_rating + F.col("sentiment_weight")) / 2, 2)
    )

    # Derive a final sentiment label from the composite score
    final_label = (
        F.when(F.col("composite_score") >= 0.25, "Positive")
        .when(F.col("composite_score") <= -0.25, "Negative")
        .otherwise("Neutral")
    )

    return enriched.withColumn("sentiment_label", final_label).select(
        "review_id",
        "booking_id",
        "user_id",
        "property_id",
        "rating",
        "comment",
        "review_date",
        "ai_sentiment",
        "composite_score",
        "sentiment_label",
    )
