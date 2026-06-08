#!/bin/bash

SCRIPT_NAME=$(basename "$0")
OUTPUT_FILE="output.txt"

# Файлы, которые нужно игнорировать везде
IGNORE_FILES=(
".gitignore"
"CITATION.cff"
"LICENSE"
"RABOTAET.ipynb"
"requirements.txt"
"TESTS_FOR_BASELINE.py"
"repo_processer.sh"
"uv.lock"
"narepy.ipynb"
"r3d_to_point_cloud.ipynb"
"lol.ipynb"
)

# Функция проверки игнорируемого файла
is_ignored_file() {
    local name="$1"
    for ignored in "${IGNORE_FILES[@]}"; do
        if [[ "$name" == "$ignored" ]]; then
            return 0
        fi
    done
    return 1
}

# Функция для обработки директории
process_directory() {
    local dir="$1"
    local indent="$2"
    
    echo "[Обработка] Директория: $dir"
    echo "${indent}Это директория $dir со следующим содержимым:" >> "$OUTPUT_FILE"
    
    while IFS= read -r -d '' entry; do
        local name=$(basename "$entry")
        local relative_path="${dir#./}/$name"
        relative_path=${relative_path#/}

        # Игнорируем сам скрипт
        if [ "$relative_path" = "$SCRIPT_NAME" ]; then
            # echo "[Пропуск] Игнорируем сам скрипт: $SCRIPT_NAME"
            continue
        fi

        # Игнорируем output.txt
        if [ "$name" = "$OUTPUT_FILE" ]; then
            # echo "[Пропуск] Игнорируем файл вывода: $OUTPUT_FILE"
            continue
        fi

        # Игнорируем файлы из списка
        if is_ignored_file "$name"; then
            # echo "[Пропуск] Игнорируем файл: $relative_path"
            continue
        fi

        if [ -f "$entry" ]; then
            echo "[Обработка] Файл: $relative_path"
            echo "${indent}Это файл $name: \`\`\`" >> "$OUTPUT_FILE"
            sed "s/^/${indent}/" "$entry" >> "$OUTPUT_FILE"
            echo "${indent}\`\`\`" >> "$OUTPUT_FILE"

        elif [ -d "$entry" ]; then
            if [ "$name" = ".git" ]; then
                # echo "[Пропуск] Игнорируем всю директорию .git"
                continue
            fi
            if [ "$name" = "data" ]; then
                # echo "[Пропуск] Игнорируем всю директорию data"
                continue
            fi
            if [ "$name" = "outputs" ]; then
                # echo "[Пропуск] Игнорируем всю директорию outputs"
                continue
            fi
            if [ "$name" = "notebooks" ]; then
                # echo "[Пропуск] Игнорируем всю директорию outputs"
                continue
            fi
            if [ "$name" = "saved" ]; then
                # echo "[Пропуск] Игнорируем всю директорию saved"
                continue
            fi
            if [ "$name" = "wandb" ]; then
                # echo "[Пропуск] Игнорируем всю директорию wandb"
                continue
            fi
            if [ "$name" = "__pycache__" ]; then
                # echo "[Пропуск] Игнорируем всю директорию __pycache__"
                continue
            fi
            if [ "$name" = ".venv" ]; then
                # echo "[Пропуск] Игнорируем всю директорию .venv"
                continue
            fi

            process_directory "$entry" "${indent}    "
        fi

    done < <(find "$dir" -mindepth 1 -maxdepth 1 -print0 | sort -z)
}

> "$OUTPUT_FILE"

echo "Начало обработки репозитория..."
echo "Игнорируем: $OUTPUT_FILE, $SCRIPT_NAME, .git/"
process_directory "." ""
echo "Обработка завершена. Результат сохранён в $OUTPUT_FILE"