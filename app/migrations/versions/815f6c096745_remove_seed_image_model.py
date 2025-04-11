"""remove_seed_image_model

Revision ID: 815f6c096745
Revises: 303b5f54be01
Create Date: 2025-04-07 22:40:36.235215

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '815f6c096745'
down_revision: Union[str, None] = '303b5f54be01'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
