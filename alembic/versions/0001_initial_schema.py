"""Create normalized hotel prediction schema."""

import sqlalchemy as sa

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("role IN ('owner', 'admin')", name="ck_users_role"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "hotels",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("location", sa.String(160), nullable=False),
        sa.Column("photo_url", sa.String(500), nullable=True),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("owner_id"),
    )
    op.create_index("ix_hotels_owner_id", "hotels", ["owner_id"], unique=True)

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("version", sa.String(64), nullable=False, unique=True),
        sa.Column("artifact_sha256", sa.String(64), nullable=False, unique=True),
        sa.Column("artifact_path", sa.String(500), nullable=False),
        sa.Column("feature_schema", sa.JSON(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_model_versions_is_active", "model_versions", ["is_active"])
    op.create_index("ix_model_versions_name_active", "model_versions", ["name", "is_active"])

    op.create_table(
        "observations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("hotel_id", sa.Integer(), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("actual_outcome", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "actual_outcome IN (0, 1) OR actual_outcome IS NULL", name="ck_actual_binary"
        ),
        sa.ForeignKeyConstraint(["hotel_id"], ["hotels.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_observations_hotel_created", "observations", ["hotel_id", "created_at"])

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("observation_id", sa.Integer(), nullable=False, unique=True),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("predicted_outcome", sa.Integer(), nullable=False),
        sa.Column("cancellation_probability", sa.Float(), nullable=True),
        sa.Column("is_correct", sa.Boolean(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("predicted_outcome IN (0, 1)", name="ck_prediction_binary"),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
        sa.ForeignKeyConstraint(["observation_id"], ["observations.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_predictions_model_created", "predictions", ["model_version_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_table("predictions")
    op.drop_table("observations")
    op.drop_table("model_versions")
    op.drop_table("hotels")
    op.drop_table("users")
