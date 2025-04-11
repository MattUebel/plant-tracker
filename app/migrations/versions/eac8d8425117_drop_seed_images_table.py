"""drop_seed_images_table

Revision ID: eac8d8425117
Revises: cf6010dfe3ae
Create Date: 2025-04-07 22:44:37.009336

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "eac8d8425117"
down_revision: Union[str, None] = "cf6010dfe3ae"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the seed_images table as we've fully migrated to using the Image model
    op.drop_table("seed_images")


def downgrade() -> None:
    # Recreate the seed_images table if needed for rollback
    op.create_table(
        "seed_images",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("seed_id", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("file_path", sa.String(length=255), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=True),
        sa.Column("mime_type", sa.String(length=100), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("ocr_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["seed_id"], ["seeds.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
