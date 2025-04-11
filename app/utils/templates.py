from fastapi.templating import Jinja2Templates
from markupsafe import Markup
import datetime
import os

# Create templates object
templates = Jinja2Templates(directory="templates")

# Add custom filters to templates
def to_date(value):
    """Convert value to date object
    
    Handles both:
    - ordinal integers (to be converted via fromordinal)
    - ISO format date strings (to be parsed)
    """
    if not value:
        return None
        
    if isinstance(value, int):
        return datetime.date.fromordinal(value)
    elif isinstance(value, str):
        try:
            # Try to parse ISO format date string (YYYY-MM-DD)
            return datetime.date.fromisoformat(value)
        except ValueError:
            # If parsing fails, return None
            return None
    return value if isinstance(value, datetime.date) else None

def strftime(date, format="%Y-%m-%d"):
    """Format date with strftime"""
    if date:
        return date.strftime(format)
    return ""

def nl2br(value):
    """Convert newlines to <br> tags"""
    if not value:
        return ""
    return Markup(value.replace('\n', '<br>'))

# Register filters
templates.env.filters["to_date"] = to_date
templates.env.filters["strftime"] = strftime
templates.env.filters["nl2br"] = nl2br