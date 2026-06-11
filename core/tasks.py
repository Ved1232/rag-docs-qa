# ─────────────────────────────────────────────────────────────────
# core/tasks.py — Celery Async Task Queue
#
# WHAT THIS FILE DOES:
#   Defines background tasks that run outside the main Streamlit process.
#   The main use case: document ingestion runs asynchronously so the
#   UI never freezes, even for large PDFs.
#
# HOW CELERY WORKS IN THIS APP:
#   There are three components:
#
#   1. PRODUCER (app.py):
#      When a user uploads a document, instead of calling ingest_text()
#      directly, we call ingest_document_task.delay(). This pushes a
#      message onto the Redis queue and returns immediately.
#      The UI is unblocked — the user can keep using the app.
#
#   2. BROKER (Redis):
#      Redis acts as the message queue between the producer and worker.
#      It stores the task messages until a worker picks them up.
#      WHY REDIS AS BROKER?
#      "Redis is already in our stack for caching. Using it as both
#       a cache and a message broker means one less infrastructure
#       component to manage. For higher volume, RabbitMQ would be
#       a more robust broker choice."
#
#   3. WORKER (separate process):
#      A Celery worker runs as a separate process (defined in docker-compose).
#      It watches the Redis queue, picks up tasks, and executes them.
#      Multiple workers can run in parallel for horizontal scaling.
#
# HOW TO START THE WORKER LOCALLY:
#   celery -A core.tasks worker --loglevel=info
#
# HOW TO START WITH DOCKER:
#   docker-compose up  (worker service defined in docker-compose.yml)
#
# TASK STATES:
#   PENDING  → task is queued, not started yet
#   STARTED  → worker has picked it up
#   SUCCESS  → completed successfully, result available
#   FAILURE  → something went wrong, error available
#
# INTERVIEW ANSWER — "Why Celery over threading?":
#   "Python's GIL (Global Interpreter Lock) means threads don't give
#    true parallelism for CPU-bound work. Celery uses separate processes
#    so workers run truly in parallel. It also gives us task monitoring,
#    retry logic, and rate limiting out of the box — things you'd have
#    to build manually with threads."
# ─────────────────────────────────────────────────────────────────

import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# ── CELERY APP CONFIGURATION ──────────────────────────────────────
# First arg: the name of the module (used for task naming)
# broker_url: where Celery sends task messages (Redis)
# result_backend: where Celery stores task results (Redis)
#
# WHY THE SAME REDIS FOR BOTH?
# "Redis supports both pub/sub messaging (for the broker) and
#  key-value storage (for results). Using one Redis instance for
#  both simplifies the infrastructure. In production with high
#  throughput, you'd use separate Redis instances to avoid
#  contention between the broker and result store."
REDIS_URL = (
    f"redis://{os.getenv('REDIS_HOST', 'localhost')}:"
    f"{os.getenv('REDIS_PORT', '6379')}/"
    f"{os.getenv('REDIS_DB', '0')}"
)

celery_app = Celery(
    'rag_tasks',
    broker=REDIS_URL,
    backend=REDIS_URL
)

# ── CELERY SETTINGS ───────────────────────────────────────────────
celery_app.conf.update(
    # How long task results are kept in Redis (seconds)
    # After this, the result is deleted to free memory
    # 3600 = 1 hour — enough time for the UI to poll for completion
    result_expires=3600,

    # Serialize tasks and results as JSON
    # WHY JSON not pickle?
    # "pickle can execute arbitrary Python code if an attacker
    #  can inject malicious pickled data. JSON is safe — it only
    #  supports strings, numbers, lists, and dicts."
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],

    # Timezone for task scheduling
    timezone='Europe/Dublin',
    enable_utc=True,

    # Retry failed tasks automatically
    # task_acks_late=True means the task is only marked as complete
    # after it finishes — if the worker dies mid-task, it retries
    task_acks_late=True,

    # Maximum number of tasks a worker processes before restarting
    # Prevents memory leaks in long-running workers
    worker_max_tasks_per_child=100,
)


# ── TASK DEFINITIONS ──────────────────────────────────────────────

@celery_app.task(
    bind=True,           # gives access to self (the task instance)
    max_retries=3,       # retry up to 3 times on failure
    default_retry_delay=10,  # wait 10 seconds between retries
    name='tasks.ingest_document'
)
def ingest_document_task(self, text, source_name, user_namespace="default"):
    """
    Background task: ingest a document into the vector store.

    WHY A BACKGROUND TASK FOR INGESTION?
    Large documents (100+ pages) take 30-60 seconds to chunk and embed.
    Running this synchronously blocks the entire Streamlit UI.
    By pushing it to Celery, the UI returns immediately and the user
    gets a progress indicator while the background worker processes.

    HOW TO CALL THIS FROM app.py:
        # Instead of calling ingest_text() directly:
        task = ingest_document_task.delay(text, source_name, namespace)
        task_id = task.id  # save this to check progress later

    HOW TO CHECK PROGRESS FROM app.py:
        from celery.result import AsyncResult
        result = AsyncResult(task_id)
        if result.state == 'SUCCESS':
            chunks_added = result.result['chunks_added']

    Args:
        self: the Celery task instance (for retry/state updates)
        text: raw document text to ingest
        source_name: label for this document (shown in citations)
        user_namespace: the user's namespace for document isolation

    Returns:
        dict: {'status': 'success', 'chunks_added': N, 'source': name}
    """
    try:
        # Import here to avoid circular imports at module level
        # and to ensure imports happen in the worker process context
        from core.ingest import ingest_text

        # Update task state so the UI can show progress
        # META dict can contain any info the UI needs
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'Processing document...',
                'source': source_name,
                'progress': 10
            }
        )

        # Run the actual ingestion
        # This is the expensive operation that used to block the UI
        chunks_added, total_chunks = ingest_text(
            text,
            source_name=source_name,
        )

        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'Indexing complete',
                'source': source_name,
                'progress': 100
            }
        )

        return {
            'status': 'success',
            'chunks_added': chunks_added,
            'total_chunks': total_chunks,
            'source': source_name,
            'namespace': user_namespace
        }

    except Exception as exc:
        # If something goes wrong, retry up to max_retries times
        # countdown=10 means wait 10 seconds before retrying
        raise self.retry(exc=exc, countdown=10)


@celery_app.task(
    name='tasks.ingest_file_task',
    max_retries=3,
    default_retry_delay=10
)
def ingest_file_task(file_path, user_namespace="default"):
    """
    Background task: ingest a file from disk into the vector store.

    Used when processing uploaded PDF files that need to be saved
    to disk first (required by PyPDFLoader).

    Args:
        file_path: absolute path to the file on disk
        user_namespace: the user's namespace for document isolation

    Returns:
        dict with ingestion results
    """
    try:
        from core.ingest import ingest_file
        import os

        # Get filename for the source label
        source_name = os.path.basename(file_path)

        chunks_added, total_chunks = ingest_file(file_path)

        # Clean up the temp file after ingestion
        if os.path.exists(file_path):
            os.unlink(file_path)

        return {
            'status': 'success',
            'chunks_added': chunks_added,
            'total_chunks': total_chunks,
            'source': source_name,
            'namespace': user_namespace
        }

    except Exception as exc:
        # Clean up temp file even on failure
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception:
                pass
        raise exc


def get_task_status(task_id):
    """
    Returns the current status of a background task.

    Called by app.py to poll for task completion and show
    progress to the user.

    HOW POLLING WORKS IN app.py:
        # Store task_id in session state
        st.session_state.pending_task = task_id

        # On each rerun (Streamlit reruns every few seconds),
        # check the status:
        status = get_task_status(st.session_state.pending_task)
        if status['state'] == 'SUCCESS':
            st.success(f"Indexed {status['result']['chunks_added']} chunks")
        elif status['state'] == 'PROGRESS':
            st.progress(status['meta']['progress'])
        elif status['state'] == 'FAILURE':
            st.error("Ingestion failed")

    Args:
        task_id: the task ID returned by .delay()

    Returns:
        dict with state, meta/result, and error if failed
    """
    from celery.result import AsyncResult

    result = AsyncResult(task_id, app=celery_app)

    if result.state == 'PENDING':
        return {
            'state': 'PENDING',
            'meta': {'status': 'Task queued, waiting for worker...', 'progress': 0}
        }

    elif result.state == 'PROGRESS':
        return {
            'state': 'PROGRESS',
            'meta': result.info  # the dict passed to update_state
        }

    elif result.state == 'SUCCESS':
        return {
            'state': 'SUCCESS',
            'result': result.result  # the dict returned by the task
        }

    elif result.state == 'FAILURE':
        return {
            'state': 'FAILURE',
            'error': str(result.info)  # the exception that was raised
        }

    else:
        return {
            'state': result.state,
            'meta': {}
        }