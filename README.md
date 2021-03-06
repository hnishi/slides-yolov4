# slides-yolov4

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="クリエイティブ・コモンズ・ライセンス" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />この 作品 は <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">クリエイティブ・コモンズ 表示 4.0 国際 ライセンス</a>の下に提供されています。

ML 論文輪読資料

To see the slide, go to

https://hnishi.github.io/slides-yolov4

## Getting Started

See https://github.com/jedcn/reveal-ck/wiki/Getting-Started

Simply,

1. Install reveal-ck
1. Create slides in slide.md
1. Generate html by `reveal-ck generate`
1. Open a browser to your slides at slides/index.html
1. `reveal-ck serve` provides a webserver on http://localhost:10000

## Publishing Slides

See https://github.com/jedcn/reveal-ck/wiki/Publishing-Slides

Use gh-pages branch

masterブランチの .gitignore に slides/ を登録

```
git checkout --orphan gh-pages  # 親commitがないgh-pagesブランチを作成。
git commit --allow-empty  # 空コミット作ってpush
git clone <git url> --branch gh-pages --single-branch ./slides  # slides配下にgh-pagesのみをclone
reveal-ck generate  # slides/ ファイルを作成
cd slides  # commit & push したらそれがgh-pagesに反映される
```

https://sue445.hatenablog.com/entry/2015/10/03/201241

## Tips

- To see presentation mode, press `s` while you're viewing the slides

- To change the path of image to public, do in vim

```vim
:%s;images/\(.*\).png;https://raw.githubusercontent.com/hnishi/slides-dotfiles/master/images/\1.png;gc
```

## References

https://github.com/jedcn/reveal-ck

https://sue445.hatenablog.com/entry/2015/10/03/201241

http://jedcn.github.io/reveal-ck/tutorial/
