"""remove_growth_field_from_seed

Revision ID: f257797328a7
Revises: 43b5d748807b
Create Date: 2025-04-10 22:54:36.286134

"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "f257797328a7"
down_revision: Union[str, None] = "43b5d748807b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("seeds", "growth")


def downgrade() -> None:
    op.add_column("seeds", sa.Column("growth", sa.Text(), nullable=True))
