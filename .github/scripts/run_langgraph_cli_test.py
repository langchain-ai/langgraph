import logging
import pathlib
import sys
import time
from urllib import error, request

import langgraph_cli
import langgraph_cli.config
import langgraph_cli.docker
from langgraph_cli.cli import prepare_args_and_stdin
from langgraph_cli.constants import DEFAULT_PORT
from langgraph_cli.exec import Runner, subp_exec
from langgraph_cli.progress import Progress

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def test(config: pathlib.Path, port: int, tag: str, verbose: bool):
    """Spin up API with Postgres/Redis via docker compose and wait until ready."""
    logger.info("Starting test...")
    with Runner() as runner, Progress(message="Pulling...") as set:
        # Detect docker/compose capabilities
        capabilities = langgraph_cli.docker.check_capabilities(runner)

        # Validate config and prepare compose stdin/args using built image
        config_json = langgraph_cli.config.validate_config_file(config)
        args, stdin = prepare_args_and_stdin(
            capabilities=capabilities,
            config_path=config,
            config=config_json,
            docker_compose=None,
            port=port,
            watch=False,
            debugger_port=None,
            debugger_base_url=f"http://127.0.0.1:{port}",
            postgres_uri=None,
            api_version=None,
            image=tag,
            base_image=None,
        )

        # Compose up with wait (implies detach), similar to `langgraph up --wait`
        args_up = [*args, "up", "--remove-orphans", "--wait"]

        compose_cmd = ["docker", "compose"]
        if capabilities.compose_type == "standalone":
            compose_cmd = ["docker-compose"]

        set("Starting...")
        try:
            runner.run(
                subp_exec(
                    *compose_cmd,
                    *args_up,
                    input=stdin,
                    verbose=verbose,
                )
            )
        except Exception as e:  # noqa: BLE001
            # On failure, show diagnostics then ensure clean teardown
            sys.stderr.write(f"docker compose up failed: {e}\n")
            try:
                sys.stderr.write("\n== docker compose ps ==\n")
                runner.run(
                    subp_exec(*compose_cmd, *args, "ps", input=stdin, verbose=False)
                )
            except Exception:
                pass
            try:
                sys.stderr.write("\n== docker compose logs (api) ==\n")
                runner.run(
                    subp_exec(
                        *compose_cmd,
                        *args,
                        "logs",
                        "langgraph-api",
                        input=stdin,
                        verbose=False,
                    )
                )
            except Exception:
                pass
            finally:
                try:
                    runner.run(
                        subp_exec(
                            *compose_cmd,
                            *args,
                            "down",
                            "-v",
                            "--remove-orphans",
                            input=stdin,
                            verbose=False,
                        )
                    )
                finally:
                    raise

        set("")
        base_url = f"http://localhost:{port}"
        ok_url = f"{base_url}/ok"
        logger.info(f"Waiting for {ok_url} to respond with 200...")
        deadline = time.time() + 30
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                with request.urlopen(ok_url, timeout=2) as resp:
                    if resp.status == 200:
                        sys.stdout.write(
                            f"""Ready!\n- API: {base_url}\n- /ok: 200 OK\n"""
                        )
                        sys.stdout.flush()
                        break
                    else:
                        last_err = RuntimeError(f"Unexpected status: {resp.status}")
                        logger.error(f"Unexpected status: {resp.status}")
            except error.URLError as e:
                logger.error(f"URLError: {e}")
                last_err = e
            except Exception as e:  # noqa: BLE001
                logger.error(f"Exception: {e}")
                last_err = e
            time.sleep(0.5)
        else:
            logger.error("Timeout waiting for /ok to return 200")
            # Bring stack down before raising
            args_down = [*args, "down", "-v", "--remove-orphans"]
            try:
                runner.run(
                    subp_exec(
                        *compose_cmd,
                        *args_down,
                        input=stdin,
                        verbose=verbose,
                    )
                )
            finally:
                raise SystemExit(
                    f"/ok did not return 202 within timeout. Last error: {last_err}"
                )

        # Clean up: bring compose stack down to free ports for next test
        logger.info("Test succeeded. Bringing down compose stack...")
        try:
            args_down = [*args, "down", "-v", "--remove-orphans"]
            runner.run(
                subp_exec(
                    *compose_cmd,
                    *args_down,
                    input=stdin,
                    verbose=verbose,
                )
            )
            logger.info("Compose stack down. Finishing...")
        except Exception:
            logger.exception("Failed to bring down compose stack")
            pass

    logger.info("Test finished")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tag", type=str)
    parser.add_argument("-c", "--config", type=str, default="./langgraph.json")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    try:
        test(pathlib.Path(args.config), args.port, args.tag, verbose=True)
    except BaseException:
        logger.exception("Test failed")
        raise
    
    logger.info("Test execution finished")
