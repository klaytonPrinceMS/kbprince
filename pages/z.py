'''
Bibliotecas importadas:
parsel: usada para fazer parsing de HTML.
httpx: usada para fazer requisições HTTP.
os: Usada para apagar arquivo caso exista e refereinciar arquivo para abertura
webbrowser: usado para abrir o arquivo no navegador


Funções:
get_html_text(urlSite)  :Faz uma requisição do tipo get no site, retornando o html em formato de texto
get_selectores_html(urlSite)    :Faz a utilização do get_html_text() pega o retorno em texto e faz o parser convertendo em seletores css e xpath
get_noticiasGN(urlSite, tagPrincipal='.gPFEn', tagTexto='a::text', tagUrl='::attr(href)', imprimir=True)    :Percorre sites de noticia padrao Google News e armazena em noticias em uma variavel
get_noticiasBBC(urlSite, tagPrincipal='.bbc-uk8dsi', tagTexto='a::text', tagUrl='::attr(href)', imprimir=True)  :Percorre sites de noticia padrao BBC e armazena em noticias em uma variavel
gerarSite() : Gera o site com os dados dos links armazenados na variavel retorna = []
'''

import httpx, parsel, os, webbrowser

class Sites:
    def __init__(self):
        self.versao                 = '1.0.20240924'
        self.bbc                    = f'https://www.bbc.com/portuguese/topics/'
        self.bbc_brasil             = f'{self.bbc}cz74k717pw5t'
        self.bbc_internacional      = f'{self.bbc}cmdm4ynm24kt'
        self.bbc_economia           = f'{self.bbc}cvjp2jr0k9rt'
        self.bbc_saude              = f'{self.bbc}c340q430z4vt'
        self.bbc_ciencia            = f'{self.bbc}cr50y580rjxt'
        self.bbc_tecnologia         = f'{self.bbc}c404v027pd4t'


        self.gn_brasil              = f'https://news.google.com/topics/CAAqJQgKIh9DQkFTRVFvSUwyMHZNREUxWm5JU0JYQjBMVUpTS0FBUAE?hl=pt-BR&gl=BR&ceid=BR%3Apt-419'
        self.gn_internacional       = f'https://news.google.com/topics/CAAqKggKIiRDQkFTRlFvSUwyMHZNRGx1YlY4U0JYQjBMVUpTR2dKQ1VpZ0FQAQ?hl=pt-BR&gl=BR&ceid=BR%3Apt-419'
        self.gn_economia            = f'https://news.google.com/topics/CAAqKggKIiRDQkFTRlFvSUwyMHZNRGx6TVdZU0JYQjBMVUpTR2dKQ1VpZ0FQAQ?hl=pt-BR&gl=BR&ceid=BR%3Apt-419'
        self.gn_saude               = f'https://news.google.com/topics/CAAqJQgKIh9DQkFTRVFvSUwyMHZNR3QwTlRFU0JYQjBMVUpTS0FBUAE?hl=pt-BR&gl=BR&ceid=BR%3Apt-419'
        self.gn_ciencia_tecnologia  = f'https://news.google.com/topics/CAAqLAgKIiZDQkFTRmdvSkwyMHZNR1ptZHpWbUVnVndkQzFDVWhvQ1FsSW9BQVAB?hl=pt-BR&gl=BR&ceid=BR%3Apt-419'
        self.gn_entretenimento  = f'https://news.google.com/topics/CAAqKggKIiRDQkFTRlFvSUwyMHZNREpxYW5RU0JYQjBMVUpTR2dKQ1VpZ0FQAQ?hl=pt-BR&gl=BR&ceid=BR%3Apt-419'
        self.gn_esporte         = f'https://news.google.com/topics/CAAqKggKIiRDQkFTRlFvSUwyMHZNRFp1ZEdvU0JYQjBMVUpTR2dKQ1VpZ0FQAQ?hl=pt-BR&gl=BR&ceid=BR%3Apt-419'
        self.gn_principais      = f'https://news.google.com/topics/news.google.com/?hl=pt-BR&gl=BR&ceid=BR%3Apt-419'

        self.msm_jfp                = f'https://jfpnoticias.com.br/monte-santo-de-minas/'
        self.agenciaBrasil      = f'https://agenciabrasil.ebc.com.br/rss/ultimasnoticias/feed.xml'
    def versaoClasse(self, imprimir=True):
        '''
        Função por padrao imprimi a versao da classe
        :param imprimir: Define por padrao que a versao sera impressa
        :return: se param imprimir for False a função retorna uma string com a versão
        '''
        if imprimir:
            print(self.versao)
        else:
            return self.versao
class Noticias():
    def __init__(self):
        self.versao                 = '1.0.20240924'
        self.retorna                = []

    def gerarSite(self, titulo='Atualizadas'):
            '''
            Função gera site com as noticias
            :param : Não existe paramentros
            Acessa os dados da variavel "retorna = []" e gera um arquivo de nome noticia.html apresnetando o texto do titulo das noticias e disponibilizando o link
            :return: Não existe Retorno
            '''
            arquivo = fr'index.html'
            if os.path.exists(arquivo):
                os.remove(arquivo)
            html = f"""
    <!DOCTYPE html>
    <!--[if IE 8 ]><html class="no-js oldie ie8" lang="en"> <![endif]-->
    <!--[if IE 9 ]><html class="no-js oldie ie9" lang="en"> <![endif]-->
    <!--[if (gte IE 9)|!(IE)]><!--><html class="no-js" lang="en"> <!--<![endif]-->
    <head>
       	<meta charset="utf-8">
    	<title>DTI - MSM</title>
    	<meta name="PRINCE. K,B" content="">
    	<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
    	<link rel="stylesheet" href="static/css/base.css">
    	<link rel="stylesheet" href="static/css/main.css">
    	<link rel="stylesheet" href="static/css/vendor.css">
    	<script src="js/modernizr.js"></script>
    	<script src="js/pace.min.js"></script>
    	<link rel="icon" type="image/png" href="static/favicon.png">
    </head>

    <body id="top">
       	<section id="intro">
    		<div class="intro-overlay"></div>
            <div class="intro-content">
                <div class="row">
                    <div class="col-twelve">
                        <h5>PRINCE. K,B.</h5>
                        <h1>DTI - Notícias</h1>
                        <p class="intro-position">
                            <span></span>
                            <span>Pref. Municipal de Monte Santo de Minas</span>
                            <span></span>
                        </p>
                        <a class="button stroke smoothscroll" href="#about" title="">Inicio</a>
                        <a class="button stroke smoothscroll" href="#services" title="">Fim</a>
                    </div>
                </div>
            </div> 
    	</section> 

    	<section id="about">
    		<div class="row section-intro">
    			<div class="col-twelve">
    				<h5>Principais notícias</h5>
    				<h1>Tema da pesquisa: {titulo}</h1>
    			</div>
    		</div>
    		<div class="row about-content">
    	   		<div class="col-six tab-full">
    				<ul class="info-list">
    					<li>
    """

            for noticia, link in self.retorna:
                html += f"""
                            <br>
                            <span><a href="{link}" target="_blank">{noticia}</a></span>
                            <strong>-------------------------------------</strong>
                """

            html += """
    					</li>
    				</ul> 
    			</div>
            </div>

        <div class="row button-section">
        </div>
    </section> 


    <section id="services">
        <div class="row services-content">
    	    <div id="owl-slider" class="owl-carousel services-list">
    	    </div>
    	</div> 
    </section>

    <footer>
        <div class="row">
            <div class="col-six tab-full">
                <div class="copyright">
                    <span>© Copyright DTI</span>
                    <span>Design by <a href="https://twitter.com/TI_SOS_Sistemas">Klayton</a></span>
                </div>		                  
                </div>
                <div id="go-top">
                <a class="smoothscroll" title="Back to Top" href="#top"><i class="fa fa-long-arrow-up"></i></a>
            </div>
        </div>  	
    </footer>  
        <div id="preloader"> 
        <div id="loader"></div>
        </div> 
        <script src="js/jquery-2.1.3.min.js"></script>
        <script src="js/plugins.js"></script>
        <script src="js/main.js"></script>
        </body>
    </html>
    """
            with open(arquivo, 'w', encoding='utf-8') as f:
                f.write(html)
            webbrowser.open(f'file://{os.path.abspath(arquivo)}')
            self.apagarNoticias()
    def versaoClasse(self, imprimir=True):
        '''
        Função por padrao imprimi a versao da classe
        :param imprimir: Define por padrao que a versao sera impressa
        :return: se param imprimir for False a função retorna uma string com a versão
        '''
        if imprimir:
            print(self.versao)
        else:
            return self.versao
    def apagarNoticias(self):
        self.retorna = []
    def get_html_text(self, urlSite):
        '''
        Função faz um get em uma url, convert em texto e retorno o texto
        :param urlSite: URL do site que será feito o get
        :return: html no formato texto
        '''
        try:
            site_html_text = httpx.get(url=urlSite).text
            return site_html_text
        except:
            return 'erro na captura do site'
    def get_selectores_html(self, urlSite):
        '''
        Função utiliza de outra função interna da classe, que retorna o html em formato de texto, pega o texto e faz um parse e retorna o html parseado
        :param urlSite: URL do site que será feito o get
        :return: Html parseado
        '''
        site_html_parsed = parsel.Selector(text=self.get_html_text(urlSite))
        return site_html_parsed
    def get_noticiasPadrao(self, urlSite, tagPrincipal='a', tagTexto='::text', tagUrl='::attr(href)', imprimir=True):
        '''
        Função acessa site Padra~p, faz o parse do html, pega uma parte do html com todos os links das noticias e entre em loop pegando o texto das noticias e o link
        :param urlSite: url do site que será parseado
        :param tagPrincipal: tag principal do site onde esta todos os itens com os links das noticias
        :param tagTexto: formato da tag onde esta o texto das noticias
        :param tagUrl: formato da tag onde esta o link
        :param imprimir: se o parametro imprimir for True o resultado da função ira salvar na variavel e ira imprimir o conteudo no terminal sem a extração dos dados
        :return: retorna para o obejeto a variavel retorna = [] com os dados obtidos no site
        '''
        seletores = self.get_selectores_html(urlSite=urlSite)
        for x in seletores.css(tagPrincipal):
            retornaTexto = f'{x.css(tagTexto).get()}'
            retornaLink  = f'{x.css(tagUrl).get()}'
            if (retornaLink not in self.retorna) and (retornaTexto not in ['Local','Página inicial','Para você','Destaques jornalísticos','Brasil','Negócios','Entreterimento','Saúde','Seguindo','Destaques Jornalísticos','Mundo','Ciência e tecnologia','Entretenimento','Esportes','','','']):
                if (retornaTexto != "None"):
                    self.retorna.append((retornaTexto, retornaLink))
            if imprimir:
                print(x)
        return self.retorna
    def get_noticiasGN(self, urlSite, tagPrincipal='.gPFEn', tagTexto='a::text', tagUrl='::attr(href)', imprimir=False):
        '''
        Função acessa site do Google News, faz o parse do html, pega uma parte do html com todos os links das noticias e entre em loop pegando o texto das noticias e o link
        :param urlSite: url do site que será parseado
        :param tagPrincipal: tag principal do site onde esta todos os itens com os links das noticias
        :param tagTexto: formato da tag onde esta o texto das noticias
        :param tagUrl: formato da tag onde esta o link
        :param imprimir: se o parametro imprimir for True o resultado da função ira salvar na variavel e ira imprimir o conteudo no terminal sem a extração dos dados
        :return: retorna para o obejeto a variavel retorna = [] com os dados obtidos no site
        '''
        seletores = self.get_selectores_html(urlSite=urlSite)
        for x in seletores.css(tagPrincipal):
            retornaTexto = f'{x.css(tagTexto).get()}'
            retornaLink  = f'https://news.google.com/{x.css(tagUrl).get()}'
            if (retornaLink not in self.retorna) and (retornaTexto not in ['Local','Página inicial','Para você','Destaques jornalísticos','Brasil','Negócios','Entreterimento',
                                                                           'Saúde','Seguindo','Destaques Jornalísticos','Mundo','Ciência e tecnologia','Entretenimento','Esportes','','','']):
                if (retornaTexto != "None"):
                    self.retorna.append((retornaTexto, retornaLink))
            if imprimir:
                print(x)
        return self.retorna
    def get_noticiasBBC(self, urlSite, tagPrincipal='.bbc-uk8dsi', tagTexto='a::text', tagUrl='::attr(href)', imprimir=False):
        '''
        Função acessa site do BBC, faz o parse do html, pega uma parte do html com todos os links das noticias e entre em loop pegando o texto das noticias e o link
        :param urlSite: url do site que será parseado
        :param tagPrincipal: tag principal do site onde esta todos os itens com os links das noticias
        :param tagTexto: formato da tag onde esta o texto das noticias
        :param tagUrl: formato da tag onde esta o link
        :param imprimir: se o parametro imprimir for True o resultado da função ira salvar na variavel e ira imprimir o conteudo no terminal sem a extração dos dados
        :return: retorna para o obejeto a variavel retorna = [] com os dados obtidos no site
        '''
        seletores = self.get_selectores_html(urlSite=urlSite)
        for x in seletores.css(tagPrincipal):
            retornaTexto = f'{x.css(tagTexto).get()}'
            retornaLink  = f'{x.css(tagUrl).get()}'
            if (retornaLink not in self.retorna):
                if (retornaTexto != "None"):
                    self.retorna.append((retornaTexto, retornaLink))
            if imprimir:
                print(x)
        return self.retorna
    def get_noticiasJFP(self, urlSite, tagPrincipal='.td-module-title a', tagTexto='a::text', tagUrl='::attr(href)', imprimir=False):
        '''
        Função acessa site Padra~p, faz o parse do html, pega uma parte do html com todos os links das noticias e entre em loop pegando o texto das noticias e o link
        :param urlSite: url do site que será parseado
        :param tagPrincipal: tag principal do site onde esta todos os itens com os links das noticias
        :param tagTexto: formato da tag onde esta o texto das noticias
        :param tagUrl: formato da tag onde esta o link
        :param imprimir: se o parametro imprimir for True o resultado da função ira salvar na variavel e ira imprimir o conteudo no terminal sem a extração dos dados
        :return: retorna para o obejeto a variavel retorna = [] com os dados obtidos no site
        '''
        seletores = self.get_selectores_html(urlSite=urlSite)
        for x in seletores.css(tagPrincipal):
            retornaTexto = f'{x.css(tagTexto).get()}'
            retornaLink  = f'{x.css(tagUrl).get()}'
            if (retornaLink not in self.retorna) and (retornaTexto not in ['Local','Página inicial','Para você','Destaques jornalísticos','Brasil','Negócios','Entreterimento','Saúde','Seguindo','Destaques Jornalísticos','Mundo','Ciência e tecnologia','Entretenimento','Esportes','','','']):
                if (retornaTexto != "None"):
                    self.retorna.append((retornaTexto, retornaLink))
            if imprimir:
                try:
                    print(retornaTexto, retornaLink)
                except:
                    print(x)

        return self.retorna










