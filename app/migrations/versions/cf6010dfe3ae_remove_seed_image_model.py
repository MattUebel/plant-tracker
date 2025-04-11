"""remove_seed_image_model

Revision ID: cf6010dfe3ae
Revises: 815f6c096745
Create Date: 2025-04-07 22:41:02.831208

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cf6010dfe3ae'
down_revision: Union[str, None] = '815f6c096745'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
