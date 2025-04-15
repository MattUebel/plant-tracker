"""remove growth field from seeds table

Revision ID: 20250415_remove_growth_from_seed
Revises: f257797328a7
Create Date: 2025-04-15 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20250415_remove_growth_from_seed"
down_revision = "f257797328a7"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_column("seeds", "growth")


def downgrade():
    op.add_column("seeds", sa.Column("growth", sa.Text(), nullable=True))
