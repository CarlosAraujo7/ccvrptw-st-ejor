@echo off
rem Ciclos B3/B4 (diário Entrada 118). Abre o lote numa janela própria do PowerShell.
rem Lançado pelo Explorer, fica fora da árvore de processos do app do Claude (19-20/10/2026).
cd /d "%~dp0.."
powershell -NoExit -ExecutionPolicy Bypass -File "%~dp0rodar_ciclo_b3b4.ps1"
