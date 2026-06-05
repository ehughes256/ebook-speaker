from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("reader", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="processedbook",
            name="status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("processing", "Processing"),
                    ("done", "Done"),
                    ("failed", "Failed"),
                ],
                db_index=True,
                default="pending",
                max_length=20,
            ),
        ),
    ]
