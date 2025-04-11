"""create_plantings_table

Revision ID: 1a5d8e4b9f12
Revises: 998185dcef1b
Create Date: 2025-03-24 14:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1a5d8e4b9f12'
down_revision: Union[str, None] = '998185dcef1b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create plantings table
    op.create_table('plantings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('seed_id', sa.Integer(), nullable=True),
        sa.Column('expected_germination_time', sa.Integer(), nullable=True),
        sa.Column('actual_germination_time', sa.Integer(), nullable=True),
        sa.Column('expected_maturity_time', sa.Integer(), nullable=True),
        sa.Column('actual_maturity_time', sa.Integer(), nullable=True),
        sa.Column('transplant_events', sa.JSON(), nullable=True),
        sa.Column('seeds_planted', sa.Integer(), nullable=True),
        sa.Column('successful_plants', sa.Integer(), nullable=True),
        sa.Column('planting_date', sa.Date(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['seed_id'], ['seeds.id'], name='fk_planting_seed_id', ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_plantings_id'), 'plantings', ['id'], unique=False)
    
    # Add planting_id column to images table
    op.add_column('images',
        sa.Column('planting_id', sa.Integer(), nullable=True)
    )
    op.create_foreign_key(
        'fk_image_planting_id',
        'images', 'plantings',
        ['planting_id'], ['id'],
        ondelete='CASCADE'
    )


def downgrade() -> None:
    # Remove foreign key constraint first
    op.drop_constraint('fk_image_planting_id', 'images', type_='foreignkey')
    
    # Remove planting_id column from images table
    op.drop_column('images', 'planting_id')
    
    # Drop plantings table
    op.drop_index(op.f('ix_plantings_id'), table_name='plantings')
    op.drop_table('plantings')