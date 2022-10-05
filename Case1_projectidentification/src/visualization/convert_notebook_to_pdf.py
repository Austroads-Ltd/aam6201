# Code modified from https://github.com/betatim/notebook-as-pdf
# BSD 3-Clause License

# Copyright (c) 2020, Tim Head <betatim@gmail.com>
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import asyncio
import tempfile
import nbformat
import os

from pyppeteer import launch
from PyPDF2 import PdfFileWriter, PdfFileReader
from argparse import ArgumentParser
from nbconvert import HTMLExporter


def parse_arguments():
    """
    Parse arguments given to this script
    - notebook_path: path to the notebook to be converted
    - output_dir: output directory
    """
    parser = ArgumentParser()
    parser.add_argument('notebook_path', type=str)
    parser.add_argument('--output-dir', type=str, default='.')
    args = parser.parse_args()
    return args

async def html_to_pdf(html_file: str, pdf_file: str, pyppeteer_args:list=None):
    """Convert a HTML file to a PDF"""
    browser = await launch(
        handleSIGINT=False,
        handleSIGTERM=False,
        handleSIGHUP=False,
        args=pyppeteer_args or [],
    )

    page = await browser.newPage()
    await page.emulateMedia("screen")
    await page.setViewport({'width': 1024, 'height': 768})
    await page.goto(f"file://{html_file}", {"waitUntil": ["networkidle2"]})


    page_margins = {
        "left": "0px",
        "right": "0px",
        "top": "20px",
        "bottom": "0px",
    }

    dimensions = await page.evaluate(
        """() => {
        return {
            width: document.body.scrollWidth,
            height: document.body.scrollHeight,
            offsetWidth: document.body.offsetWidth,
            offsetHeight: document.body.offsetHeight,
            deviceScaleFactor: window.devicePixelRatio,
        }
    }"""
    )
    
    width = dimensions['width']
    height = dimensions["height"]
    print_width = min(width + 2, 200 * 72)
    print_height = min(height + 2, 200 * 72)
    print(f"Print width: {print_width}; print height: {print_height}")
    print(f"Chromium version: {await browser.version()}")



    await page.evaluate(
        """
        var style = document.createElement('style');
        style.innerHTML = `
            #notebook-container {
                box-shadow: none;
                padding: unset
            }
            div.cell {
                page-break-inside: avoid;
            }
            div.output_wrapper {
                page-break-inside: avoid;
            }
            div.output {
                page-break-inside: avoid;
            }
            /* Jupyterlab based HTML uses these classes */
            .jp-Cell-inputWrapper {
                page-break-inside: avoid;
            }
            .jp-Cell-outputWrapper {
                page-break-inside: avoid;
            }
            .jp-Notebook {
                margin: 0px;
            }
            /* Hide the message box used by MathJax */
            #MathJax_Message {
                display: none;
            }
            pre {
                white-space: pre-wrap;
            } 
        `
        document.head.appendChild(style);
        """
    )
    
    await page.evaluate(
        """
        function getOffset( el ) {
            var _x = 0;
            var _y = 0;
            while( el && !isNaN( el.offsetLeft ) && !isNaN( el.offsetTop ) ) {
                _x += el.offsetLeft - el.scrollLeft;
                _y += el.offsetTop - el.scrollTop;
                el = el.offsetParent;
            }
            return { top: _y, left: _x };
            }
        """,
        force_expr=True,
    )

    # script for breaking up large table into smaller chunks
    await page.evaluate(
        """
        function getRightEdgeX(elem) {
            var boundingRec = elem.getBoundingClientRect();
            return boundingRec.x + boundingRec.width;
        }

        function split_down(table) {
            var cols = table.getElementsByTagName('thead').item(0).getElementsByTagName('th');
            var rows = table.getElementsByTagName('tbody').item(0).getElementsByTagName('tr');
            var parent = table.parentNode;

            // make new table
            var newTable = document.createElement('table');
            var newTableHead = document.createElement('thead');
            var newTableBody = document.createElement('tbody');
            newTable.setAttribute('class', 'dataframe');
            newTable.appendChild(newTableHead);
            newTable.appendChild(newTableBody);

            // create empty table rows
            for (var i = 0; i < rows.length; i++) {
                var newTR = document.createElement('tr');
                newTableBody.appendChild(newTR);
            }
            var new_rows = newTableBody.getElementsByTagName('tr');

            // populate rows and removing existing table's columns
            var i = cols.length - 1;
            while (getRightEdgeX(table) > document.body.scrollWidth) {
                // append outer most column. Without copying, this also removes existing table's column
                newTableHead.insertBefore(cols.item(i), newTableHead.firstChild);
                for (var j=0; j < new_rows.length; j++) {
                    var r = new_rows.item(j);
                    r.insertBefore(rows.item(j).lastElementChild, r.firstChild);
                }
                i -= 1;
            }

            // append index
            newTableHead.insertBefore(document.createElement('th'), newTableHead.firstChild);
            for (var i = 0; i < rows.length; i++) {
                var r = new_rows.item(i);
                var index = rows.item(i).getElementsByTagName('th').item(0).cloneNode(true);
                r.insertBefore(index, r.firstChild);
            }

            parent.appendChild(document.createElement('br'));
            parent.appendChild(newTable);

            return newTable;
        }

        var dataframes = document.querySelectorAll('table.dataframe');
        for (var i=0; i < dataframes.length; i++) {
            var df = dataframes.item(i);
            while (getRightEdgeX(df) > document.body.scrollWidth) {
                df = split_down(df); 
            }
        }
        """,
        force_expr=True
    )

    # with open('/Users/hoanglongdang/Desktop/debug.html', 'w') as f:
    #     f.write(await page.content())

    await page.pdf(
        {
            "path": pdf_file,
            # Adobe can not display pages longer than 200inches. So we limit
            # ourselves to that and start a new page if needed.
            "width": print_width,
            "height": print_height,
            "printBackground": True,
            "margin": page_margins,
        }
    )

    headings = await page.evaluate(
        """() => {
        var vals = []
        for (const elem of document.getElementsByTagName("h1")) {
            vals.push({ top: getOffset(elem).top * (1-72/288), text: elem.innerText })
        }
        for (const elem of document.getElementsByTagName("h2")) {
            vals.push({ top: getOffset(elem).top * (1-72/288), text: "∙ " + elem.innerText })
        }
        return vals
    }"""
    )

    await browser.close()

    return headings


def finish_pdf(pdf_in: str, pdf_out: str, notebook: dict, headings: list):
    """Add finishing touches to the PDF file.

    To make the PDF nicer we:

    * attach the original notebook to the PDF for reference
    * add bookmarks pointing to the headers in a notebook
    """
    pdf = PdfFileWriter()
    pdf.appendPagesFromReader(PdfFileReader(pdf_in, "rb"))
    pdf.addAttachment(notebook["file_name"], notebook["contents"])

    for heading in sorted(headings, key=lambda x: x["top"]):
        page_num = int(heading["top"]) // (200 * 72)

        page_height = pdf.getPage(page_num).artBox[-1]

        # position on the page as measured from the bottom of the page
        # with a bit of leeway so that clicking the bookmark doesn't put
        # the heading right at the border
        on_page_pos = page_height - (int(heading["top"]) % (200 * 72)) + 20

        # there is no nice way of passing the "zoom arguments" at the very
        # end of the function call without explicitly listing all the parameters
        # of the function. We can't use keyword arguments :(
        pdf.addBookmark(
            heading["text"],
            page_num,
            None,
            None,
            False,
            False,
            "/XYZ",
            0,
            on_page_pos,
            None,
        )

    with open(pdf_out, "wb") as fp:
        pdf.write(fp)


async def notebook_to_pdf(html_notebook: str, pdf_path: str, pyppeteer_args: list=None):
    """ Wrapper over html_to_pdf for writing html string into a file """
    with tempfile.NamedTemporaryFile(suffix=".html") as f:
        f.write(html_notebook.encode())
        f.flush()
        heading_positions = await html_to_pdf(f.name, pdf_path, pyppeteer_args)

    return heading_positions

def convert_notebook():
    """ Convert notebook.ipynb to notebook.pdf """
    args = parse_arguments()
    notebook = nbformat.read(args.notebook_path, as_version=4)
    output_path = os.path.join(args.output_dir, os.path.basename(args.notebook_path).replace(".ipynb", ".pdf"))

    html_exporter = HTMLExporter()
    html_notebook, _ = html_exporter.from_notebook_node(notebook, resources=None)

    with tempfile.TemporaryDirectory(suffix="nb-as-pdf") as name:
        pdf_fname = os.path.join(name, "output.pdf")
        pdf_fname2 = os.path.join(name, "output-with-attachment.pdf")
        pyppeteer_args = ["--no-sandbox"]

        heading_positions = asyncio.run(
            notebook_to_pdf(html_notebook, pdf_fname, pyppeteer_args=pyppeteer_args)
        )

        finish_pdf(
            pdf_fname,
            pdf_fname2,
            {
                "file_name": os.path.basename(args.notebook_path),
                "contents": nbformat.writes(notebook).encode("utf-8"),
            },
            heading_positions,
        )

        with open(pdf_fname2, "rb") as f:
            pdf_bytes = f.read()

        # output into output_path
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

if __name__ == '__main__':
    convert_notebook()