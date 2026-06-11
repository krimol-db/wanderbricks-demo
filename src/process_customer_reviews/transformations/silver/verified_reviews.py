from pyspark import pipelines as dp


@dp.materialized_view(
    comment="Reviews from verified customers who completed a booking with a successful payment"
)
def verified_reviews():
    reviews = spark.read.table("samples.wanderbricks.reviews")
    bookings = spark.read.table("samples.wanderbricks.bookings")
    payments = spark.read.table("samples.wanderbricks.payments")

    # Only keep reviews that are not deleted
    active_reviews = reviews.filter("is_deleted = false")

    # Join reviews with bookings that are confirmed or completed
    verified = active_reviews.join(
        bookings.filter("status IN ('confirmed', 'completed')").select("booking_id"),
        on="booking_id",
        how="inner",
    )

    # Further filter to only bookings with a completed payment
    verified_with_payment = verified.join(
        payments.filter("status = 'completed'").select("booking_id").distinct(),
        on="booking_id",
        how="inner",
    )

    return verified_with_payment.select(
        "review_id",
        "booking_id",
        "user_id",
        "property_id",
        "rating",
        "comment",
        verified_with_payment["created_at"].alias("review_date"),
    )
