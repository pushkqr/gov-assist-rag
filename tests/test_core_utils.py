import os
import unittest
from unittest.mock import MagicMock, patch

from core.utils import get_aistudio_client, get_genai_client, embed_content_safe


class CoreUtilsTests(unittest.TestCase):
    def test_get_genai_client_vertex_mode(self):
        with patch.dict(os.environ, {"USE_VERTEX_AI": "True", "GOOGLE_CLOUD_PROJECT": "p1", "GOOGLE_CLOUD_LOCATION": "loc1"}):
            with patch("core.utils.genai.Client") as mock_client:
                get_genai_client()
                mock_client.assert_called_once_with(vertexai=True, project="p1", location="loc1")

    def test_embed_content_safe_routes_to_aistudio_when_key_present(self):
        vertex_client = MagicMock()
        mock_aistudio_client = MagicMock()

        with patch.dict(os.environ, {"USE_AISTUDIO_FOR_EMBEDDINGS": "True", "GEMINI_API_KEY": "test_api_key"}):
            with patch("core.utils.get_aistudio_client", return_value=mock_aistudio_client):
                embed_content_safe(vertex_client, model="gemini-embedding-001", contents="test text")
                mock_aistudio_client.models.embed_content.assert_called_once_with(
                    model="gemini-embedding-001", contents="test text"
                )
                vertex_client.models.embed_content.assert_not_called()


if __name__ == "__main__":
    unittest.main()
