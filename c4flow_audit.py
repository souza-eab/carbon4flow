"""
c4flow_audit.py
───────────────
Módulo de auditoria de acesso do Carbon4Flow.

Responsabilidades:
  - Anonimizar email e IP via HMAC-SHA256 com salt (st.secrets)
  - Criptografar email em texto claro via Fernet AES-256 (st.secrets)
  - Gerar session_id único por sessão
  - Inserir eventos na tabela BigQuery access_events
  - Expor funções simples para o app: init_session, log_event

Tabela destino:
  ee-souza759.carbon4flow_audit.access_events

Secrets necessários em .streamlit/secrets.toml:
  [audit]
  salt        = "string-secreta-longa-e-aleatoria"
  encrypt_key = "chave-fernet-gerada-com-Fernet.generate_key()"

Eventos possíveis:
  'login'      → disparado na barreira de entrada após consentimento
  'heartbeat'  → disparado periodicamente para medir tempo ativo
  'logout'     → disparado se o usuário encerrar explicitamente
  'timeout'    → disparado quando a sessão expira por inatividade

Notas LGPD:
  - Email e IP nunca são armazenados em texto claro
  - HMAC-SHA256 com salt torna os hashes irreversíveis sem o salt
  - email_enc é criptografado com Fernet — só decifrável com encrypt_key
  - encrypt_key armazenada apenas em st.secrets, nunca no repositório
  - Retenção automática de 90 dias via partition_expiration_days no BigQuery

CHANGELOG:
  - Adicionado campo email_enc (Fernet AES-256) para identificação admin
  - project ID corrigido para ee-souza759
"""

import hmac
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import streamlit as st
from cryptography.fernet import Fernet
from google.cloud import bigquery
from google.oauth2 import service_account

# ─────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────

_PROJECT     = "ee-souza759"
_TABLE       = f"{_PROJECT}.carbon4flow_audit.access_events"
_APP_VERSION = "0.0.4"  # atualizar a cada deploy

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Cliente BigQuery — reutiliza credenciais GCP
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _bq_client() -> bigquery.Client:
    """
    Reutiliza o service account já configurado para o GCS.
    Adiciona o escopo BigQuery às credenciais.
    """
    info  = dict(st.secrets["gcp_service_account"])
    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/bigquery",
        ],
    )
    return bigquery.Client(credentials=creds, project=_PROJECT)


# ─────────────────────────────────────────────
# Anonimização — HMAC-SHA256
# ─────────────────────────────────────────────

def _hmac_hash(value: str) -> str:
    """
    HMAC-SHA256(value, salt).

    Mais seguro que SHA-256(value + salt) pois o HMAC
    usa o salt como chave criptográfica — resistente a
    ataques de extensão de comprimento (length extension attacks).

    Retorna string hex de 64 chars.
    """
    salt = st.secrets["audit"]["salt"].encode("utf-8")
    return hmac.new(salt, value.encode("utf-8"), hashlib.sha256).hexdigest()


def hmac_hash(value: str) -> str:
    """Versão pública do hash para uso externo (ex: aba de privacidade)."""
    return _hmac_hash(value)


# ─────────────────────────────────────────────
# Criptografia — Fernet AES-256
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _fernet() -> Fernet:
    """
    Instância Fernet usando encrypt_key dos secrets.
    cache_resource garante instância única por worker.
    """
    key = st.secrets["audit"]["encrypt_key"].encode("utf-8")
    return Fernet(key)


def _encrypt_email(email: str) -> str:
    """
    Criptografa o email com Fernet AES-256.
    Retorna string base64 — armazenável como STRING no BigQuery.
    Só decifrável com a encrypt_key correta.
    """
    return _fernet().encrypt(email.encode("utf-8")).decode("utf-8")


def decrypt_email(token: str) -> str:
    """
    Decifra um email_enc da tabela BigQuery.

    USO LOCAL — rode no script decrypt_emails.py, nunca no app em produção.

    Parâmetro:
      token : valor do campo email_enc vindo do BigQuery

    Retorna o email em texto claro.
    Levanta InvalidToken se a chave estiver errada ou o token corrompido.
    """
    return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")


# ─────────────────────────────────────────────
# Captura de contexto
# ─────────────────────────────────────────────

def _get_ip() -> str:
    """
    Tenta capturar o IP real do usuário.
    No Streamlit Cloud o IP real vem via X-Forwarded-For.
    Retorna 'unknown' se não conseguir — nunca quebra o fluxo.
    """
    try:
        headers   = st.context.headers
        forwarded = headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return headers.get("Remote-Addr", "unknown")
    except Exception:
        return "unknown"


def _get_user_agent() -> str:
    """Captura o User-Agent do navegador."""
    try:
        return st.context.headers.get("User-Agent", "unknown")
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────
# Gestão de sessão
# ─────────────────────────────────────────────

def init_session(email: str) -> str:
    """
    Inicializa a sessão do usuário no st.session_state.

    - Gera session_id único (UUID v4)
    - Armazena hash do email (nunca o email em texto claro)
    - Criptografa email para gravação no BigQuery
    - Registra timestamp de início
    - Dispara evento 'login' no BigQuery

    Deve ser chamada uma única vez, logo após o consentimento
    e validação do email na barreira de entrada.

    Retorna o session_id gerado.
    """
    session_id = str(uuid.uuid4())

    st.session_state["session_id"]     = session_id
    st.session_state["hash_email"]     = _hmac_hash(email)
    st.session_state["email_enc"]      = _encrypt_email(email)
    st.session_state["session_start"]  = datetime.now(timezone.utc)
    st.session_state["last_heartbeat"] = datetime.now(timezone.utc)
    st.session_state["autenticado"]    = True

    log_event("login")
    return session_id


def get_session_duration() -> Optional[int]:
    """
    Retorna duração da sessão em segundos desde o login.
    Retorna None se a sessão não foi inicializada.
    """
    start = st.session_state.get("session_start")
    if start is None:
        return None
    return int((datetime.now(timezone.utc) - start).total_seconds())


# ─────────────────────────────────────────────
# Insert BigQuery
# ─────────────────────────────────────────────

def log_event(evento: str) -> bool:
    """
    Insere um evento de auditoria na tabela BigQuery.

    Parâmetros:
      evento : 'login' | 'heartbeat' | 'logout' | 'timeout'

    Retorna True se o insert foi bem-sucedido, False caso contrário.
    Nunca levanta exceção — falha silenciosa para não quebrar o app.
    """
    try:
        now        = datetime.now(timezone.utc)
        hash_email = st.session_state.get("hash_email", "unknown")
        email_enc  = st.session_state.get("email_enc", None)
        session_id = st.session_state.get("session_id", "unknown")
        duracao    = get_session_duration() if evento in ("logout", "timeout") else None

        row = {
            "timestamp_utc":    now.isoformat(),
            "event_date":       now.date().isoformat(),
            "hash_email":       hash_email,
            "email_enc":        email_enc,
            "hash_ip":          _hmac_hash(_get_ip()),
            "session_id":       session_id,
            "evento":           evento,
            "duracao_segundos": duracao,
            "user_agent":       _get_user_agent(),
            "app_version":      _APP_VERSION,
        }

        errors = _bq_client().insert_rows_json(_TABLE, [row])

        if errors:
            log.error(f"[audit] BigQuery insert_rows_json errors: {errors}")
            return False

        return True

    except Exception as e:
        log.error(f"[audit] Falha ao logar evento '{evento}': {e}")
        return False


# ─────────────────────────────────────────────
# Exclusão de dados — Art. 18 LGPD
# ─────────────────────────────────────────────

def delete_user_data(email: str) -> tuple[bool, int]:
    """
    Remove todos os registros do usuário da tabela BigQuery.

    Recebe o email em texto claro, gera o hash internamente
    e executa DELETE por hash — o email nunca é usado na query.

    Retorna (sucesso: bool, linhas_removidas: int).
    """
    try:
        hash_email = _hmac_hash(email)

        query = f"""
            DELETE FROM `{_TABLE}`
            WHERE hash_email = @hash_email
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("hash_email", "STRING", hash_email)
            ]
        )

        job = _bq_client().query(query, job_config=job_config)
        job.result()

        rows_deleted = job.num_dml_affected_rows or 0
        log.info(f"[audit] LGPD delete: {rows_deleted} registros removidos para hash {hash_email[:8]}...")
        return True, rows_deleted

    except Exception as e:
        log.error(f"[audit] Falha ao executar delete LGPD: {e}")
        return False, 0
