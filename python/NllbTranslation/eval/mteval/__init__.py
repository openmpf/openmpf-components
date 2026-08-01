"""Modules behind the NllbTranslation evaluation harness.

Nothing here is meant to be invoked directly by a user -- the shell scripts in the
parent directory are the entry points. These modules are split across **two
runtimes**, and mixing them up produces failures that only appear at run time:

Host runtime -- the scoring venv built by ``../setup_venv.sh``. Invoked as
``./venv/bin/python3 -m mteval.<module>`` from the ``eval/`` directory:

* ``tmx_sample``    extract a fixed-seed sample from a TMX
* ``make_sample``   the same, from Moses-format parallel files
* ``mt_eval``       BLEU/chrF/chrF++/TER + COMET + paired bootstrap
* ``summarize``     combine per-pair metrics into ``results/SUMMARY.md``
* ``decomp_report`` render one decomposition run as Markdown

Container runtime -- run *inside* an OpenMPF NllbTranslation image, where
``eval/`` is bind-mounted at ``/eval``. Invoked by absolute path, e.g.
``/opt/mpf/plugin-venv/bin/python /eval/mteval/ct2_driver.py``:

* ``nllb_eval_driver``  one translation per input line, via the component
* ``ct2_driver``        CTranslate2 directly, so a specific model and
                        ``compute_type`` can be forced
* ``bench_split_mode``  in-process throughput benchmark

The two sets share no imports, and must not. The host modules depend on
``numpy``/``sacrebleu``/``comet``, none of which exist in the component images;
the container modules depend on ``mpf_component_api``/``nllb_component``/
``ctranslate2``, none of which are in the scoring venv. There is deliberately no
shared helper module -- anything common would have to satisfy both, and the
duplication is cheaper than that constraint.
"""
