# encoding: utf-8

import os
import jinja2
from jaqs.util import fileio
# from weasyprint import HTML


class Report(object):
    def __init__(self, dic, source_dir, template_fn, out_folder='.'):
        """

        Parameters
        ----------
        dic : dict
        source_dir : str
            path of directory where HTML template and css files are stored.
        template_fn : str
            File name of HTML template.
        out_folder : str
            Output folder of report.

        """
        self.dic = dic
        self.template_fn = template_fn
        self.out_folder = os.path.abspath(out_folder)

        self.env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=[source_dir]))
        self._update_env()

        self.template = self.env.get_template(self.template_fn)
        self.html = ""

    def _update_env(self):
        """Define custom functions we use in HTML template."""
        def round_if_float(x, n):
            if isinstance(x, float):
                return round(x, n)
            else:
                return x
        self.env.filters.update({'round_if_float': round_if_float})
    
    def generate_html(self):
        self.html = self.template.render(self.dic)
    
    def output_html(self, fn='test_out.html'):
        path = os.path.abspath(os.path.join(self.out_folder, fn))
        
        fileio.create_dir(path)
        with open(path, 'w') as f:
            f.write(self.html)

        print "HTML report: {:s}".format(path)
    
    def output_pdf(self, fn='test_out.html'):
        pass
        # h = HTML(string=self.html)
        # h.write_pdf(os.path.join(self.out_folder, fn))#, stylesheets=[self.fp_css])
