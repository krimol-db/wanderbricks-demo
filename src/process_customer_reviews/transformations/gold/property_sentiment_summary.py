from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.materialized_view(
    comment="Sentiment reviews enriched with property and destination information for geographic analysis"
)
def property_sentiment_summary():
    sentiment = spark.read.table("sentiment_reviews")
    properties = spark.read.table("samples.wanderbricks.properties")
    destinations = spark.read.table("samples.wanderbricks.destinations")

    # Join sentiment reviews with property details
    with_property = sentiment.join(
        properties.select(
            "property_id",
            "title",
            "property_type",
            "destination_id",
            "property_latitude",
            "property_longitude",
        ),
        on="property_id",
        how="inner",
    )

    # Join with destinations for location info
    with_location = with_property.join(
        destinations.select("destination_id", "destination", "country", "state_or_province"),
        on="destination_id",
        how="inner",
    )

    return with_location.select(
        "review_id",
        "property_id",
        "title",
        "property_type",
        "destination",
        "country",
        "state_or_province",
        "property_latitude",
        "property_longitude",
        "rating",
        "comment",
        "ai_sentiment",
        "composite_score",
        "sentiment_label",
        "review_date",
    )
