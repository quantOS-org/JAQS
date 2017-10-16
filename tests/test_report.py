# encoding: utf-8
from jaqs.trade.analyze.report import Report
from jaqs.util import fileio


def test_output():
    static_folder = fileio.join_relative_path('trade/analyze/static')

    r = Report({'mytitle': 'Test Title', 'mytable': 'Hello World!'},
               source_dir=static_folder,
               template_fn='test_template.html',
               out_folder='../output')
    r.generate_html()
    r.output_html()
    r.output_pdf()


if __name__ == "__main__":
    test_output()
