# IEEE conference report: run latexmk from this directory, or use -cd main.tex.
$pdf_mode = 1;
$out_dir = 'build';
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error %O %S';
$bibtex_use = 1;
$max_repeat = 5;
$sleep_time = 1;
$view = 'none';
$preview_continuous_mode = 0;

# GUI applications on macOS may not inherit the TeX installation's PATH.
if ($^O eq 'darwin' && -d '/Library/TeX/texbin') {
    $ENV{'PATH'} = '/Library/TeX/texbin:' . $ENV{'PATH'};
}
