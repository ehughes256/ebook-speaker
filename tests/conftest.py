import pytest
from django.conf import settings


@pytest.fixture
def settings_fixture(db, settings):
    """Fixture to provide access to Django settings."""
    return settings
